from itertools import count
import queue
import time
import logging
from functools import reduce
from collections import deque
from argparse import Namespace
import torch
import torch.multiprocessing as mp

if torch.cuda.is_available():
    mp.set_start_method('spawn')

import numpy as np
from inferno.io.transform import Compose
from inferno.io.transform.generic import Normalize
from inferno.io.transform.image import ElasticTransform, RandomFlip, RandomRotate

logger = logging.getLogger('Trainy')


class Trainer(object):
    def __init__(self, handler, hyperparameters=None):
        # Privates
        self._handler = handler
        # Preprocessing
        self._raw_preprocessor = None
        self._joint_preprocessor = None
        # Training
        self._data_queue: mp.Queue = None
        self._abort_event: mp.Event = None
        self._pause_event: mp.Event = None
        self._training_process: mp.Process = None
        self._ignited = False
        # Publics
        # Sane default hparams
        if hyperparameters is None:
            self.hparams = Namespace(optimizer_kwargs=dict(lr=0.0003, weight_decay=0.0001,
                                                           amsgrad=True),
                                     optimizer_name='Adam',
                                     criterion_kwargs=dict(reduce=False),
                                     criterion_name='BCEWithLogitsLoss',
                                     batch_size=1,
                                     cache_size=8)
        else:
            self.hparams: Namespace = hyperparameters

    @property
    def model(self):
        return self._handler.model

    def share_memory(self):
        self._handler._model = self._handler._model.share_memory()
        return self

    @property
    def device(self):
        return self._handler.device

    @staticmethod
    def _train_process(model: torch.nn.Module,
                       device: torch.device,
                       data_queue: mp.Queue,
                       abort: mp.Event,
                       pause: mp.Event,
                       hparams: Namespace):
        logger.info(f"Initializing Loss and Optimizer.")
        # Set up what's needed for training
        criterion = getattr(torch.nn, hparams.criterion_name)(**hparams.criterion_kwargs)
        optim = getattr(torch.optim, hparams.optimizer_name)(model.parameters(), **hparams.optimizer_kwargs)
        # Init a cache. In case there are not enough batches in data_queue,
        # we'll use it to top up the batch with what's in this cache.
        data_cache = deque(maxlen=hparams.cache_size)
        while True:
            # Init a batch
            batch = []
            # Check if abort event is set
            if abort.is_set():
                logger.info(f"Aborting...")
                break
            if pause.is_set():
                logger.info(f"Waiting for resume...")
                time.sleep(1)
            try:
                sample = 0
                while len(batch) < hparams.batch_size:
                    logger.info(f"Trying to Fetch sample {sample} of {hparams.batch_size}...")
                    # Try to fetch from data queue
                    data, labels, weights = data_queue.get(block=False)
                    logger.info(f"Fetched sample {sample} of {hparams.batch_size}...")
                    # Add to batch
                    batch.append((data, labels, weights))
                    # Add to cache
                    data_cache.append((data, labels, weights))
                    sample += 1
            except queue.Empty:
                logger.info(f"Queue Exhausted.")
                if len(batch) == 0 and len(data_cache) == 0:
                    # Both batch and cache empty, try again
                    logger.info(f"Trying to fetch again...")
                    time.sleep(0.1)
                    continue
                elif len(batch) == hparams.batch_size:
                    # Nothing to do here
                    pass
                elif len(batch) < hparams.batch_size:
                    # Batch not full, try to top it up from the cache
                    logger.info(f"Topping up batch, currently with {len(batch)} elements...")
                    while len(data_cache) > 0 and len(batch) < hparams.batch_size:
                        data_sample = data_cache.popleft()
                        batch.append(data_sample)
                        data_cache.append(data_sample)
                else:
                    logger.error(f"LOLWTF: len(batch) = {len(batch)}, "
                                  f"len(data_cache) = {len(data_cache)}")
                    raise RuntimeError
            logger.info(f"Updating with {len(batch)} samples...")
            # Make a batch
            data, labels, weights = zip(*batch)
            logger.debug(f"data.shapes = {[list(t.shape) for t in data]}, "
                          f"label.shapes = {[list(t.shape) for t in labels]}, "
                          f"weights.shapes = {[list(t.shape) for t in weights]}")
            data, labels, weights = (torch.stack(data, dim=0),
                                     torch.stack(labels, dim=0),
                                     torch.stack(weights, dim=0))
            # Ship tensors to device
            data, labels, weights = data.to(device), labels.to(device), weights.to(device)
            logger.info(f"Transferred to device.")
            # Train the model
            prediction = model(data)
            logger.info(f"Fed forward.")
            loss = criterion(prediction, labels).mul(weights).mean()
            logger.info(f"Loss Evaluated.")
            optim.zero_grad()
            loss.backward()
            logger.info(f"Backproped.")
            optim.step()
            logger.info(f"Stepped.")

    def ignition(self):
        # Done in this method:
        #   1. Init data queue
        #   2. Init abort event
        #   3. Start the training process
        logger.info("Prepping Queue and Event...")
        self._data_queue = mp.Queue()
        self._abort_event = mp.Event()
        self._pause_event = mp.Event()
        logger.info("Sharing Memory...")
        self.share_memory()
        self._training_process = mp.Process(target=self._train_process,
                                            args=(self.model, self.device,
                                                  self._data_queue, self._abort_event,
                                                  self._pause_event, self.hparams))
        logger.info("3, 2, 1...")
        self._training_process.start()
        logger.info("We have lift off.")
        self._ignited = True

    def shut_down_training_process(self):
        if self._training_process is not None:
            # Shut down the training process
            logger.info("Setting Abort Event...")
            self._abort_event.set()
            for trial in range(6):
                logger.info(f"Try {trial} of 5:")
                # Give training process some time to die
                if self._training_process.is_alive():
                    logger.info(f"Process Alive.")
                    time.sleep(10)
                else:
                    break
            logger.info(f"Process Dead.")

    def __del__(self):
        # Shut down the training process
        self.shut_down_training_process()
        self._ignited = False

    def _preprocess(self, data, labels):
        # labels.shape = data.shape = (c, z, y, x)
        # FIXME Not have these hard coded
        if self._raw_preprocessor is None:
            self._raw_preprocessor = Normalize()
        if self._joint_preprocessor is None:
            self._joint_preprocessor = Compose(RandomFlip(),
                                               RandomRotate(),
                                               ElasticTransform(alpha=2000., sigma=50.))
        # Convert data and labels to torch tensors
        with torch.no_grad():
            # Apply transforms
            data = self._raw_preprocessor(data)
            data, labels = self._joint_preprocessor(data, labels)
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            # Obtain weight map
            weights = labels.gt(0)
            # Label value 0 actually corresponds to Ignore. Subtract 1 from all pixels that will be
            # weighted to account for that
            labels[weights] -= 1
        # Done
        return data, labels, weights.float()

    def ensure_ignited(self):
        if not self._ignited:
            logger.info("Ignition...")
            self.ignition()

    def push(self, data, labels):
        # Done in this method:
        #   1. Augment data
        #   2. Push to queue
        self.ensure_ignited()
        logger.info(f"Feeding {len(data)} samples to queue...")
        # Augment
        for _data, _labels in zip(data, labels):
            _data, _labels, _weights = self._preprocess(_data, _labels)
            self._data_queue.put((_data, _labels, _weights))
        logger.info(f"Fed {len(data)} samples to queue...")

    def pause(self):
        logger.info("Pausing training...")
        self._pause_event.set()

    def resume(self):
        logger.info("Resuming training...")
        self._pause_event.clear()

import torch
from inspect import signature
from inferno.utils import torch_utils as thu
from inferno.trainers.basic import Trainer


class TikTorchTrainer(Trainer):
    def __init__(self, model=None):
        super(TikTorchTrainer, self).__init__(model)

    def train_for(self, num_iterations=None, break_callback=None):
        # Switch model to train mode
        self.train_mode()
        # Call callback
        self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_RUN, num_iterations=num_iterations)
        # iteration_num is a local clock. There's the global self._iteration_count that keeps
        # actual track of the number of iterations - this is updated by the call to
        # self.next_iteration().
        iteration_num = 0
        while True:
            if num_iterations is not None and iteration_num >= num_iterations:
                self.console.info("Finished {} iterations. Breaking...".format(num_iterations))
                break
            # Break if break callback asks us to
            if break_callback is not None and break_callback(iteration_num):
                self.console.info("Breaking on request from callback.")
                break
            self.console.progress(
                "Training iteration {} (batch {} of epoch {}).".format(
                    iteration_num, self._batch_count, self._epoch_count
                )
            )
            # Call callback
            self.callbacks.call(self.callbacks.BEGIN_OF_TRAINING_ITERATION, iteration_num=iteration_num)
            # Zero out the grads
            self.optimizer.zero_grad()

            # Get batch
            batch = self.fetch_next_batch("train")
            # Send to device and wrap as variable
            batch = self.wrap_batch(batch, from_loader="train")
            # Separate inputs from targets
            inputs, target = self.split_batch(batch, from_loader="train")
            # Apply model, compute loss and backprop
            prediction, loss = self.apply_model_and_loss(inputs, target, backward=True, mode="train")
            # Compute metric
            if self.metric_is_defined and self.evaluate_metric_now:
                self._last_metric_evaluated_at_epoch = self._epoch_count
                error = self.metric(thu.unwrap(prediction, to_cpu=False), thu.unwrap(target, to_cpu=False))
                self.update_state("training_error", thu.unwrap(error))
            else:
                error = None
            # Update state from computation
            #            self.update_state('training_inputs', thu.unwrap(inputs))
            #            self.update_state('training_target', thu.unwrap(target))
            #            self.update_state('training_prediction', thu.unwrap(prediction))
            #            self.update_state('training_loss', thu.unwrap(loss))
            # Update state from model's state hooks
            self.update_state_from_model_state_hooks()
            # Update parameters
            self.optimizer.step()
            # Call callback
            self.callbacks.call(self.callbacks.END_OF_TRAINING_ITERATION, iteration_num=iteration_num)
            # Prepare for next iteration
            self.next_iteration()
            # Break if validating or saving. It's important that the next_iteration() method is
            # called before checking validate_now and save_now - because otherwise, the iteration
            # counter is never updated after the first save and validate, resulting in an infinite
            # save + validate loop.
            if self.validate_now:
                self.console.info("Breaking to validate.")
                break
            if self.save_now:
                self.console.info("Breaking to save.")
                break
            iteration_num += 1

        self.callbacks.call(self.callbacks.END_OF_TRAINING_RUN, num_iterations=num_iterations)
        return self

    def apply_model_and_loss(self, inputs, target, backward=True, mode=None):
        if mode is None:
            mode = self._current_mode
            assert_(
                mode in ["train", "eval"], f"`mode` must be one of ['train', 'eval'], got {mode} instead.", ValueError
            )
        # Compute prediction
        prediction = self.apply_model(*inputs)

        # Mask
        mask = target.gt(0)
        target = torch.masked_select(target, mask) - 1
        prediction = torch.masked_select(prediction, mask)

        # Compute loss
        kwargs = {}
        if isinstance(self.criterion, torch.nn.Module) and "trainer" in signature(self.criterion.forward).parameters:
            kwargs["trainer"] = self
        if mode == "train":
            loss = self.criterion(prediction, target, **kwargs)
        elif mode == "eval":
            loss = self.validation_criterion(prediction, target, **kwargs)
        else:
            raise ValueError
        if backward:
            # Backprop if required
            # retain_graph option is needed for some custom
            # loss functions like malis, False per default
            loss.backward(retain_graph=self.retain_graph)
        return prediction, loss

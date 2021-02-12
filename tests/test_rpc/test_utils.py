import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tiktorch.rpc.utils import BatchedExecutor


def test():
    blocks = list(range(100))

    def process_block(block_id):
        time.sleep(random.random() * 0.1)
        if random.random() < 0.2:
            raise Exception()
        return (block_id,)

    futures = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        batcher = BatchedExecutor(batch_size=20)

        for id_ in blocks:

            def work(*a, **kw):
                return ex.submit(process_block, *a, **kw)

            futures.append(batcher.submit(work, id_))

        count = 0
        for f in as_completed(futures):
            count += 1

    assert count == len(blocks)

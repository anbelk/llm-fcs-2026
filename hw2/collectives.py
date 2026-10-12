import torch
import torch.distributed as dist


def _exchange(send, receive, next_rank, previous_rank):
    requests = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send.contiguous(), next_rank),
        dist.P2POp(dist.irecv, receive, previous_rank),
    ])
    for request in requests:
        request.wait()


def ring_reduce_scatter(tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if tensor.ndim != 1 or tensor.numel() % world_size:
        raise ValueError("Длина одномерного тензора должна делиться на число процессов")
    chunks = [chunk.clone() for chunk in tensor.chunk(world_size)]
    previous_rank, next_rank = (rank - 1) % world_size, (rank + 1) % world_size
    for step in range(world_size - 1):
        send_index = (rank - step - 1) % world_size
        receive_index = (rank - step - 2) % world_size
        received = torch.empty_like(chunks[receive_index])
        _exchange(chunks[send_index], received, next_rank, previous_rank)
        chunks[receive_index].add_(received)
    return chunks[rank]


def ring_all_gather(shard):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    chunks = [torch.empty_like(shard) for _ in range(world_size)]
    chunks[rank].copy_(shard)
    previous_rank, next_rank = (rank - 1) % world_size, (rank + 1) % world_size
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        _exchange(chunks[send_index], chunks[receive_index], next_rank, previous_rank)
    return torch.cat(chunks)


def ring_all_reduce(tensor):
    return ring_all_gather(ring_reduce_scatter(tensor))

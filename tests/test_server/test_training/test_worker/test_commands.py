from tiktorch.server.training.worker import commands as cmds


class TestCommandQueue:
    def test_stop_command_has_higher_priorityj(self):
        cmd_queue = cmds.CommandPriorityQueue()
        stop_cmd = cmds.StopCmd()
        cmd_queue.put_nowait(cmds.ResumeCmd())
        cmd_queue.put_nowait(stop_cmd)
        cmd_queue.put_nowait(cmds.PauseCmd())

        received_cmd = cmd_queue.get_nowait()
        assert stop_cmd is received_cmd

    def test_queue_order_is_stable(self):
        cmd_queue = cmds.CommandPriorityQueue()
        stop_cmds = [cmds.StopCmd() for _ in range(100)]
        for cmd in stop_cmds:
            cmd_queue.put_nowait(cmd)

        for expected_cmd in stop_cmds:
            assert expected_cmd is cmd_queue.get_nowait()

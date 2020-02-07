from tiktorch.server.training.worker import commands as cmds


class TestCommandQueue:
    def test_shop_command_has_high_priority(self):
        cmd_queue = cmds.CommandPriorityQueue()
        stop_cmd = cmds.StopCmd()
        cmd_queue.put_nowait(cmds.ResumeCmd())
        cmd_queue.put_nowait(stop_cmd)
        cmd_queue.put_nowait(cmds.PauseCmd())

        received_cmd = cmd_queue.get_nowait()
        assert stop_cmd is received_cmd

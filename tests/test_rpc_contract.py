import sys

import pytest

from piperx_expo.config import ROOT
from piperx_expo.server.launch import _install_rpc_contract


def test_rpc_contract_never_retries_unknown_action(monkeypatch):
    sys.path.insert(0, str(ROOT / "third_party/expo"))
    from expo_ft.env.env_client import EnvClient, EnvClientWrapper

    # install只在正式启动进程生效；测试退出恢复原类，避免影响其他协议测试。
    for cls, names in ((EnvClient, ("create_env", "_call_operation")), (EnvClientWrapper, ("_call",))):
        for name in names:
            monkeypatch.setattr(cls, name, getattr(cls, name))
    _install_rpc_contract({})

    class BrokenConnection:
        sends = 0

        def send(self, payload):
            self.sends += 1

        def recv(self):
            raise RuntimeError("接收时断线，无法知道动作是否已执行")

    connection = BrokenConnection()
    client = EnvClient()
    monkeypatch.setattr(client, "_get_connection", lambda: connection)
    monkeypatch.setattr(client, "_close_connection", lambda: None)
    with pytest.raises(RuntimeError, match="断线"):
        client._call_operation("step", {"env_id": "test", "action": [0] * 7})
    assert connection.sends == 1
    wrapper = object.__new__(EnvClientWrapper)
    with pytest.raises(RuntimeError):
        wrapper._call("step", lambda: client._call_operation("step", {}))
    assert connection.sends == 2

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def test_awex_runtime_settings_are_forwarded_to_ray_workers(monkeypatch):
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_MAX_CHANNELS", "8")
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_HCA_POLICY", "balanced")
    monkeypatch.setenv("AWEX_NCCL_DEVICE_V2_EXTENSION", "awex_nccl_device_ext_v2")
    monkeypatch.setenv("AWEX_PROFILE", "1")
    monkeypatch.setenv("AWEX_PROFILE_SYNC_START", "1")
    monkeypatch.setenv("AWEX_PROFILE_WARMUP_UPDATES", "2")
    monkeypatch.setenv("AWEX_NCCL_MAX_OPS_PER_PEER_BATCH", "64")

    env_vars = get_ppo_ray_runtime_env()["env_vars"]

    assert env_vars["NCCL_CUMEM_ENABLE"] == "1"
    assert env_vars["AWEX_NCCL_DEVICE_V2_MAX_CHANNELS"] == "8"
    assert env_vars["AWEX_NCCL_DEVICE_V2_HCA_POLICY"] == "balanced"
    assert env_vars["AWEX_NCCL_DEVICE_V2_EXTENSION"] == "awex_nccl_device_ext_v2"
    assert env_vars["AWEX_PROFILE"] == "1"
    assert env_vars["AWEX_PROFILE_SYNC_START"] == "1"
    assert env_vars["AWEX_PROFILE_WARMUP_UPDATES"] == "2"
    assert env_vars["AWEX_NCCL_MAX_OPS_PER_PEER_BATCH"] == "64"

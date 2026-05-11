from pathlib import Path

import numpy as np
import torch

from algorithms.dqn import DQNAgent
from algorithms.maddpg import MADDPG
from evaluate import evaluate


def test_dqn_agent_save_load_round_trip_preserves_networks_and_actions():
    obs = np.zeros((3, 4), dtype=np.float32)
    source = DQNAgent(obs_dim=4, hidden_dim=16, seed=1)
    target = DQNAgent(obs_dim=4, hidden_dim=16, seed=2)
    path = Path("dqn_checkpoint_test.pt")

    with torch.no_grad():
        for param in source.network.parameters():
            param.add_(0.5)
        for param in source.target_network.parameters():
            param.sub_(0.25)

    try:
        source.save(path)
        target.load(path)

        for source_param, target_param in zip(source.network.parameters(), target.network.parameters()):
            torch.testing.assert_close(source_param, target_param)
        for source_param, target_param in zip(source.target_network.parameters(), target.target_network.parameters()):
            torch.testing.assert_close(source_param, target_param)
        np.testing.assert_array_equal(source.act(obs, epsilon=0.0)[1], target.act(obs, epsilon=0.0)[1])
    finally:
        path.unlink(missing_ok=True)


def test_dqn_agent_load_rejects_missing_network_keys():
    agent = DQNAgent(obs_dim=4, hidden_dim=16, seed=1)
    path = Path("bad_dqn_checkpoint_test.pt")

    try:
        torch.save({"network": agent.network.state_dict()}, path)
        try:
            agent.load(path)
        except ValueError as exc:
            assert "target_network" in str(exc)
        else:
            raise AssertionError("expected invalid DQN checkpoint to raise ValueError")
    finally:
        path.unlink(missing_ok=True)


def test_maddpg_save_load_round_trip_preserves_deterministic_actions():
    obs = np.zeros((2, 4), dtype=np.float32)
    source = MADDPG(num_users=2, obs_dim=4, action_dim=3, hidden_dim=16, seed=3)
    target = MADDPG(num_users=2, obs_dim=4, action_dim=3, hidden_dim=16, seed=4)
    path = Path("maddpg_checkpoint_test.pt")

    with torch.no_grad():
        for agent in source.agents:
            for param in agent.actor.parameters():
                param.add_(0.1)
            for param in agent.critic.parameters():
                param.add_(0.2)
            for param in agent.target_actor.parameters():
                param.add_(0.3)
            for param in agent.target_critic.parameters():
                param.add_(0.4)

    try:
        source.save(path)
        target.load(path)

        np.testing.assert_allclose(source.act(obs, noise_std=0.0), target.act(obs, noise_std=0.0))
        for source_agent, target_agent in zip(source.agents, target.agents):
            for source_param, target_param in zip(source_agent.critic.parameters(), target_agent.critic.parameters()):
                torch.testing.assert_close(source_param, target_param)
            for source_param, target_param in zip(
                source_agent.target_critic.parameters(),
                target_agent.target_critic.parameters(),
            ):
                torch.testing.assert_close(source_param, target_param)
    finally:
        path.unlink(missing_ok=True)


def test_maddpg_load_rejects_mismatched_agent_lists():
    maddpg = MADDPG(num_users=2, obs_dim=4, action_dim=3, hidden_dim=16, seed=3)
    path = Path("bad_maddpg_checkpoint_test.pt")

    try:
        torch.save(
            {
                "actors": [maddpg.agents[0].actor.state_dict()],
                "critics": [agent.critic.state_dict() for agent in maddpg.agents],
                "target_actors": [agent.target_actor.state_dict() for agent in maddpg.agents],
                "target_critics": [agent.target_critic.state_dict() for agent in maddpg.agents],
            },
            path,
        )
        try:
            maddpg.load(path)
        except ValueError as exc:
            assert "actors" in str(exc)
        else:
            raise AssertionError("expected invalid MADDPG checkpoint to raise ValueError")
    finally:
        path.unlink(missing_ok=True)


def test_evaluate_records_loaded_and_missing_checkpoint_status():
    dqn = DQNAgent(obs_dim=10, hidden_dim=128, seed=10)
    maddpg = MADDPG(num_users=6, obs_dim=10, action_dim=3, hidden_dim=128, seed=10)
    dqn_path = Path("eval_dqn_checkpoint_test.pt")
    maddpg_path = Path("eval_maddpg_checkpoint_test.pt")

    try:
        dqn.save(dqn_path)
        maddpg.save(maddpg_path)
        rows = evaluate(episodes=1, dqn_checkpoint=dqn_path, maddpg_checkpoint=maddpg_path, load_models=True)
        by_name = {row["algorithm"]: row for row in rows}

        assert by_name["DQN"]["checkpoint_loaded"] is True
        assert by_name["MADDPG"]["checkpoint_loaded"] is True
        assert by_name["All Local"]["checkpoint_status"] == "not_applicable"

        rows = evaluate(
            episodes=1,
            dqn_checkpoint=Path("missing_dqn_checkpoint_test.pt"),
            maddpg_checkpoint=Path("missing_maddpg_checkpoint_test.pt"),
            load_models=True,
        )
        by_name = {row["algorithm"]: row for row in rows}

        assert by_name["DQN"]["checkpoint_loaded"] is False
        assert by_name["DQN"]["checkpoint_status"] == "missing"
        assert by_name["MADDPG"]["checkpoint_loaded"] is False
        assert by_name["MADDPG"]["checkpoint_status"] == "missing"
    finally:
        dqn_path.unlink(missing_ok=True)
        maddpg_path.unlink(missing_ok=True)

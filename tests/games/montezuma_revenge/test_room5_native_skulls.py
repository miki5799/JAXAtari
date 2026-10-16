"""Room5 skull motion measured in ALE, without policy or future-state injection."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtari.games.montezuma_revenge.rooms import load_room

REFERENCE = json.loads((Path(__file__).parent / "fixtures/room5_native.json").read_text())

@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["label"])
def test_room5_skulls_native_ale_trajectory(montezuma_env, case):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(100001785))
    state = load_room(jnp.int32(5), state, env.consts).replace(
        # Keep the player safely on the lower ladder, like the ALE wait probe.
        player_x=jnp.int32(76), player_y=jnp.int32(127),
        is_climbing=jnp.int32(1), last_ladder=jnp.int32(0),
        native_frame_count=jnp.int32(case["entry_frame"]),
        # Deliberately unrelated pose clock: bounce must be room-local.
        frame_count=jnp.int32(999),
    )
    for frame, x, y in case["frames"]:
        obs, state, reward, done, _ = env.step(state, jnp.int32(0))
        assert int(state.native_frame_count) == frame
        np.testing.assert_array_equal(np.asarray(state.enemies_x[:2]), [x, x + 16], err_msg=f"frame{frame}")
        np.testing.assert_array_equal(np.asarray(obs.enemies.y[:2]), [y, y], err_msg=f"frame{frame}")
        assert int(state.room_id) == 5
        assert float(reward) == 0 and not bool(done)


def test_room5_reentry_resets_bounce_without_resetting_native_clock(montezuma_env):
    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(100001785))
    state = state.replace(native_frame_count=jnp.int32(2137), room5_skull_phase=jnp.int32(14))
    state = load_room(jnp.int32(5), state, env.consts)
    assert int(state.native_frame_count) == 2137
    assert int(state.room5_skull_phase) == 0
    assert bool(state.room5_skull_spawn_pending)
    np.testing.assert_array_equal(np.asarray(env._get_observation(state).enemies.y[:2]), [54, 54])

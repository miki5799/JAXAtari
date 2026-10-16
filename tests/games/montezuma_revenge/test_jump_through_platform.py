import jax
import jax.numpy as jnp
import pytest

def test_jump_through_dynamic_platform(montezuma_env):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Place player on platform at y=76 (Platform index 2 in ROOM_2_1)
    # player_y = 76 - 1 (to be on top) - 19 (height offset) = 56
    state = state.replace(
        room_id=jnp.array(17, dtype=jnp.int32),
        player_x=jnp.array(4, dtype=jnp.int32),
        player_y=jnp.array(56, dtype=jnp.int32), # Feet at 75
        fall_distance=jnp.array(0, dtype=jnp.int32),
        is_falling=jnp.array(0, dtype=jnp.int32),
        is_jumping=jnp.array(0, dtype=jnp.int32)
    )
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(17, dtype=jnp.int32), state, env.consts)

    # Ensure we are on platform 2 (y=76)
    assert state.platforms_y[2] == 76

    # Jump UP (Action 1 is FIRE, which triggers jump if on ground)
    obs, state, reward, done, info = env.step(state, 1)
    assert state.is_jumping == 1

    # We want to jump through platform 1 (y=66).
    # We start at feet=75.
    # The FIRE frame prepares the jump without moving. The subsequent frames
    # move feet to 72, 69, 66, 64 and 62, passing through the upper platform.

    for _ in range(5):
        obs, state, reward, done, info = env.step(state, 0)

    # Should be at player_y = 56 - 13 = 43 (feet at 62)
    assert state.player_y == 43
    assert state.is_jumping == 1

    # Now wait until landing. Should land on platform 1 (y=66), so player_y = 66 - 1 - 19 = 46.
    landed = False
    for _ in range(50):
        obs, state, reward, done, info = env.step(state, 0)
        if state.is_jumping == 0 and state.is_falling == 0:
            landed = True
            break

    assert landed
    assert state.player_y == 46
    assert state.player_y + 19 == 65 # Feet just above platform 1

@pytest.mark.parametrize("player_x,start_y,landing_y,mode,next_y", [
    pytest.param(4, 36, 26, 4, 26, id="left-side-step"),
    pytest.param(8, 36, 26, 4, 26, id="left-supported-edge"),
    pytest.param(9, 36, 36, 7, 38, id="left-unsupported-edge"),
    pytest.param(10, 36, 36, 7, 38, id="left-inner-anchor"),
    pytest.param(12, 36, 36, 7, 38, id="left-sprite-contact"),
    pytest.param(140, 36, 36, 7, 38, id="right-sprite-contact"),
    pytest.param(142, 36, 36, 7, 38, id="right-inner-anchor"),
    pytest.param(143, 36, 36, 7, 38, id="right-unsupported-edge"),
    pytest.param(144, 36, 26, 4, 26, id="right-supported-edge"),
    pytest.param(148, 36, 26, 4, 26, id="right-side-step"),
    pytest.param(10, 86, 86, 7, 88, id="left-through-middle-shelf"),
    pytest.param(142, 86, 86, 7, 88, id="right-through-middle-shelf"),
])
def test_jump_through_static_platform(montezuma_env, player_x, start_y, landing_y, mode, next_y):
    env = montezuma_env
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Room 17 has fixed shelves with standing anchors y=26 and y=76.
    # A side-step hop passes upward through the shelf above its start.
    state = state.replace(
        room_id=jnp.array(17, dtype=jnp.int32),
        player_x=jnp.array(player_x, dtype=jnp.int32),
        player_y=jnp.array(start_y, dtype=jnp.int32),
        fall_distance=jnp.array(0, dtype=jnp.int32),
        is_falling=jnp.array(0, dtype=jnp.int32),
        is_jumping=jnp.array(0, dtype=jnp.int32)
    )
    from jaxtari.games.montezuma_revenge.rooms import load_room
    state = load_room(jnp.array(17, dtype=jnp.int32), state, env.consts)

    # Jump UP
    obs, state, reward, done, info = env.step(state, 1)
    assert state.is_jumping == 1

    # ALE uses short hops at the supported anchors x<=8 / x>=144. Sprite
    # contact permits a full hop just inside these edges, but cannot support
    # the standing anchor after that hop finishes.
    for _ in range(20):
        obs, state, reward, done, info = env.step(state, 0)

    assert state.is_jumping == 0
    assert state.player_y == landing_y
    assert state.native_mode == mode
    if mode == 7:
        assert state.fall_distance == 0

    obs, state, reward, done, info = env.step(state, 0)
    assert state.player_y == next_y


@pytest.mark.parametrize("player_x,supported", [(8, True), (9, False), (143, False), (144, True)])
def test_stair_support_prepares_fall_before_moving(montezuma_env, player_x, supported):
    from jaxtari.games.montezuma_revenge.rooms import load_room

    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.int32(17), state, env.consts).replace(
        player_x=jnp.int32(player_x), player_y=jnp.int32(36))

    _, state, _, done, _ = env.step(state, 0)
    assert not done
    assert state.player_y == 36
    assert state.native_mode == (4 if supported else 7)

    _, state, _, done, _ = env.step(state, 0)
    assert not done
    assert state.player_y == (36 if supported else 38)


@pytest.mark.parametrize("player_x", [10, 142])
@pytest.mark.parametrize("start_y,falling_ticks,fatal", [(56, 8, True), (58, 7, False)])
def test_stair_fall_damage_boundary(montezuma_env, player_x, start_y, falling_ticks, fatal):
    from jaxtari.games.montezuma_revenge.rooms import load_room

    env = montezuma_env
    _, state = env.reset(jax.random.PRNGKey(0))
    state = load_room(jnp.int32(17), state, env.consts).replace(
        player_x=jnp.int32(player_x), player_y=jnp.int32(start_y))
    lives = state.lives
    # One support-loss tick, then two pixels per native falling tick.
    for _ in range(falling_ticks + 1):
        _, state, _, _, _ = env.step(state, 0)
    assert state.player_y == 72
    assert state.native_mode == 7
    assert state.death_timer == 0

    # B65B starts death at eight ticks, while seven still land safely.
    _, state, _, _, _ = env.step(state, 0)
    assert state.player_y == 76
    assert state.native_mode == (6 if fatal else 4)
    assert state.death_type == int(fatal)
    assert state.lives == lives - int(fatal)

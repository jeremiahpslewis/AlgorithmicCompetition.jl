import CircularArrayBuffers
using ReinforcementLearningTrajectories
using CircularArrayBuffers: CircularArrayBuffer

# Adapted from https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl/blob/b50a9f37e94b07030770943041efbc34252c290b/src/common/CircularArraySARTTraces.jl
# License: https://github.com/JuliaReinforcementLearning/ReinforcementLearningTrajectories.jl/blob/b50a9f37e94b07030770943041efbc34252c290b/LICENSE
# NOTE: This patch is necessary so that we don't log next_action, which creates trace issues (SARS q-learning algorithm doesn't require next_action)

const CircularArraySARSTraces = Traces{
    SS′ART,
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}

function CircularArraySARSTraces(;
    capacity::Int,
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
    Traces(
        action=CircularArrayBuffer{action_eltype}(action_size..., capacity),
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySARSTraces) = CircularArrayBuffers.capacity(t.traces[end])

#####
# ComposedStopCondition
#####

"""
    ComposedStopCondition(stop_conditions...; reducer = any)

The result of `stop_conditions` is reduced by `reducer`.
"""
struct ComposedStopCondition{S,T}
    stop_conditions::S
    reducer::T
    function ComposedStopCondition(stop_conditions...; reducer = any)
        new{typeof(stop_conditions),typeof(reducer)}(stop_conditions, reducer)
    end
end

function (s::ComposedStopCondition)(args...)
    s.reducer(sc(args...) for sc in s.stop_conditions)
end

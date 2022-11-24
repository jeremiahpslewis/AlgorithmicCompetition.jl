using ReinforcementLearning

function RLCore.EpsilonGreedyExplorer(beta::Rational)
	EpsilonGreedyExplorer(kind=:exp,
		ϵ_init=1,
		ϵ_stable=0,
		decay_steps=(1/beta),
	)
end

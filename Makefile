.PHONY: figures test

figures:
	python -m experiments.polynomial_sweep.figures.eta_ladder
	python -m experiments.polynomial_sweep.figures.count_sequence
	python -m experiments.polynomial_sweep.figures.sigma_window

test:
	pytest tests/ -v

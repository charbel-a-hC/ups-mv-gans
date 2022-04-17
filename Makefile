.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo "  env		prepare environment and install required dependencies"
	@echo "  clean		remove all temp files along with docker images and docker-compose networks"
	@echo "  format	reformat code"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."



.PHONY: env-docker
env-docker:
	conda env update --prune -f environment.yml
	conda activate tf2


.PHONY: format
format:
	poetry run bash scripts/format.sh

.PHONY: clean
clean: # Remove Python file artifacts
	find . -name '*.pyc' -exec rm -rf {} +
	find . -name '*.pyo' -exec rm -rf {} +
	find . -name '*~' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -fr {} +
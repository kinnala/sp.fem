SHELL:=/bin/bash

install: install-triangle ## Install all manual dependencies
	@echo "Done!"

install-triangle: ## Download and compile Triangle
	@curl -o fem/triangle/triangle.zip http://www.netlib.org/voronoi/triangle.zip
	@unzip fem/triangle/triangle.zip -d fem/triangle
	@make -C fem/triangle

dev-install: ## Create a development environment to conda
	@conda create --name spfemenv --file requirements.txt

activate: ## Type "source activate spfemenv" to start the conda environment
	@echo "Activate the development environment by typing: \"source activate spfemenv\""

deactivate: ## Type "source deactivate" to quit the conda environment
	@echo "Deactivate the development environment by typing: \"source deactivate\""

run-tests: ## Run the unit tests
	@ipython -m unittest discover ./fem

build-docs: ## Run Sphinx to build the documentation
	@make -C docs html

help: ## Show this help (default)
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

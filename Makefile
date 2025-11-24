# ===== Makefile =====
# Usage:
#   make                # builds processed data (default)
#   make raw            # runs src/get_data.py -> data/raw/combined_data.csv
#   make processed      # runs src/clean_data.py -> data/processed/cleaned_data.csv
#   make env            # create/update conda env from environment.yml
#   make update-env     # force update the env
#   make remove-env     # remove the env
#   make clean          # delete generated data
#   make help           # show this help message
#
# You can override ENV_NAME:  make ENV_NAME=myenv raw

# --- Config ---
ENV_FILE := environment.yml
# Auto-detect env name from environment.yml (first "name:" line). Override with: make ENV_NAME=...
ENV_NAME ?= $(shell awk '/^name:/ {print $$2; exit}' $(ENV_FILE))
CONDA_RUN := conda run -n $(ENV_NAME)
STAMP_DIR := .conda
STAMP := $(STAMP_DIR)/$(ENV_NAME).stamp

# Import config.yml to get data directories (requires yq)
DATA_RAW := $(shell $(CONDA_RUN) python -c "import yaml; print(yaml.safe_load(open('config.yml'))['data_directory']['raw'])")
DATA_PROCESSED := $(shell $(CONDA_RUN) python -c "import yaml; print(yaml.safe_load(open('config.yml'))['data_directory']['processed'])")
DATA_SIM := $(shell $(CONDA_RUN) python -c "import yaml; print(yaml.safe_load(open('config.yml'))['data_directory']['simulations'])")
# Default target
.PHONY: all
all: raw processed pca simulation

# Update data end-to-end
.PHONY: update-data
update-data: clean raw processed ## Wipe CSVs and pull latest data end-to-end

# Update simulation only
.PHONY: update-sim
update-sim:
	@echo ">>> Updating simulation only"
	@rm -f $(DATA_SIM)/simulated_yield_curves.csv
	$(MAKE) simulation

# --- Environment management ---
.PHONY: env
env: $(STAMP) ## Create or update the conda environment

$(STAMP): $(ENV_FILE)
	@echo ">>> Ensuring conda env '$(ENV_NAME)' matches $(ENV_FILE)"
	@mkdir -p $(STAMP_DIR)
	@# Create or update the environment
	@if conda env list | awk '{print $$1}' | grep -qx '$(ENV_NAME)'; then \
		echo ">>> Updating existing env $(ENV_NAME)"; \
		conda env update -n $(ENV_NAME) -f $(ENV_FILE); \
	else \
		echo ">>> Creating env $(ENV_NAME)"; \
		conda env create -n $(ENV_NAME) -f $(ENV_FILE); \
	fi
	@date > $(STAMP)

.PHONY: update-env
update-env: ## Force-update the env from environment.yml
	@echo ">>> Forcing update of env $(ENV_NAME)"
	conda env update -n $(ENV_NAME) -f $(ENV_FILE)
	@mkdir -p $(STAMP_DIR) && date > $(STAMP)

.PHONY: remove-env
remove-env: ## Remove the conda env (careful!)
	@echo ">>> Removing env $(ENV_NAME)"
	-conda env remove -n $(ENV_NAME)
	@rm -f $(STAMP)

# --- Data pipeline ---
# Raw data
raw: $(DATA_RAW)/combined_data.csv ## Build raw dataset

$(DATA_RAW)/combined_data.csv: src/get_data.py | env
	@echo ">>> Running get_data.py in env $(ENV_NAME)"
	$(CONDA_RUN) python src/get_data.py

# Processed data
processed: $(DATA_PROCESSED)/cleaned_data.csv ## Build processed dataset

$(DATA_PROCESSED)/cleaned_data.csv: src/clean_data.py $(DATA_RAW)/combined_data.csv | env
	@echo ">>> Running clean_data.py in env $(ENV_NAME)"
	$(CONDA_RUN) python src/clean_data.py

# --- Analysis steps ---
pca: $(DATA_PROCESSED)/pca_factors.csv ## Run PCA analysis

$(DATA_PROCESSED)/pca_factors.csv $(DATA_PROCESSED)/pca_loadings.csv $(DATA_PROCESSED)/pca_explained_variance.csv: src/pca.py $(DATA_PROCESSED)/cleaned_data_diffs.csv | env
	@echo ">>> Running pca.py in env $(ENV_NAME)"
	$(CONDA_RUN) python src/pca.py

# --- Rate Simulation ---
simulation: $(DATA_SIM)/simulated_yield_curves.csv ## Run rate simulation

$(DATA_SIM)/simulated_yield_curves.csv: src/rate_simulation.py $(DATA_PROCESSED)/cleaned_data.csv $(DATA_PROCESSED)/pca_factors.csv $(DATA_PROCESSED)/pca_loadings.csv | env
	@echo ">>> Running rate_simulation.py in env $(ENV_NAME)"
	$(CONDA_RUN) python src/rate_simulation.py

# --- Utilities ---
.PHONY: run
run: | env ## Convenience: run get_data.py directly
	$(CONDA_RUN) python src/get_data.py

.PHONY: clean
clean: ## Remove generated data
	@echo ">>> Cleaning data directories"
	@rm -rf $(DATA_RAW)/*.csv $(DATA_PROCESSED)/*.csv $(DATA_SIM)/*.csv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_/.-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'
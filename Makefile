# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* violence_detection/*.py

black:
	@black scripts/* violence_detection/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr violence_detection-*.dist-info
	@rm -fr violence_detection.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      		    API
# ----------------------------------
run_api:
	uvicorn api.fast:app --reload


# ----------------------------------
#      		GCP Set-up
# ----------------------------------

PROJECT_ID=le-wagon-bootcamp-321818

BUCKET_NAME=wagon-data-violence-detection

REGION=europe-west1

PYTHON_VERSION=3.7

FRAMEWORK=scikit-learn

RUNTIME_VERSION=1.15

FILENAME=trainer ###### CHECK THIS IS THE SAME NAME AS THE FILE

LOCAL_PATH="raw_data"

DATA_BUCKET_FOLDER=data

MODEL_BUCKET_FOLDER=model

DATA_BUCKET_FILE_NAME=$(shell basename ${DATA_PATH})

MODEL_BUCKET_FILE_NAME=$(shell basename ${MODEL_PATH})

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	@gsutil cp -R ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_model:
	-@gsutil cp ${MODEL_PATH} gs://${BUCKET_NAME}/${MODEL_BUCKETFOLDER}/${MODEL_BUCKET_FILE_NAME}


# ----------------------------------
#            GCP Online Training
# ----------------------------------

BUCKET_TRAINING_FOLDER =trainings

JOB_NAME=violence_detection$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--scale-tier BASIC_GPU \
		--region ${REGION} \
		--master-image-uri ${IMAGE_URI} \
		--stream-logs

# ----------------------------------
#            Docker Image
# ----------------------------------

DOCKER_IMAGE_NAME= docker_violence_detection

IMAGE_REPO_NAME= violence_detection

IMAGE_TAG= 602-sep-violence-detection

IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

docker_build:
	docker build -t ${IMAGE_URI} ./
docker_run:
	docker run --gpus all ${IMAGE_URI}

docker_push:
	docker push ${IMAGE_URI}

# ----------------------------------
# Download Best Model from checkpoint
# ----------------------------------

download_best:
	gsutil -m cp -r dir gs://${BUCKET_NAME}/ ###### CHANGE_THIS_VARIABLE #######
	
#### CHANGE to name of architecture used in trainer

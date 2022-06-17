all: benchmark

benchmark: docker/container.ok data_input/folding_yeast4.pickle
	docker container run --rm -it -v `(pwd)`:/benchmark/data convgenbenchmark python3 /benchmark/data/run_all_exercises.py
	docker container run --rm -it -v `(pwd)`:/benchmark/data convgenbenchmark chown -R `(./getMyUid)` /benchmark/data/data_result

benchmark-gpu: docker/container.ok data_input/folding_yeast4.pickle
	docker container run --rm --gpus all -it -v `(pwd)`:/benchmark/data convgenbenchmark python3 /benchmark/data/run_all_exercises.py
	docker container run --rm -it -v `(pwd)`:/benchmark/data convgenbenchmark chown -R `(./getMyUid)` /benchmark/data/data_result


docker/container.ok: docker/Dockerfile docker/run.sh docker/requirements.txt
	docker build -t convgenbenchmark docker/.
	date > $@


data_input/folding_yeast4.pickle:
	cd data_input && tar -xzf data.tgz
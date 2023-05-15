all: benchmark

benchmark: docker/container.ok
	docker container run --rm -it -v `(pwd)`:/benchmark/data convgenbenchmark python3 /benchmark/data/run_all_exercises.py
	make fix

benchmark-gpu: docker/container.ok
	docker container run --rm --gpus all -it -v `(pwd)`:/benchmark/data convgenbenchmark python3 /benchmark/data/run_all_exercises.py
	make fix

fix: docker/container.ok
	docker container run --rm -it -v `(pwd)`:/benchmark/data convgenbenchmark chown -R `(./getMyUid)` /benchmark/data/data_result


clean: fix
	rm -f data_result/*/folding_*.csv
	rm -f data_result/*/folding_*.log
	rm -f data_result/*/folding_*.log.time

docker/container.ok: docker/Dockerfile docker/run.sh docker/requirements.txt
	docker build -t convgenbenchmark docker/.
	date > $@


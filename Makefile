all: isort reformat test
format: isort reformat

isort:
	isort .

reformat:
	black .

test:
	pytest .
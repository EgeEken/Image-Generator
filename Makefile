.PHONY: build up logs stop clean

build:
	docker compose build

up:
	docker compose up -d

logs:
	docker compose logs -f

stop:
	docker compose down

clean:
	docker compose down --rmi all --volumes --remove-orphans

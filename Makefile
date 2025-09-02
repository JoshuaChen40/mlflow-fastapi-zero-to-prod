# 預設 docker compose 指令 (方便之後切換成 docker compose v2)
DC = docker-compose

# 啟動容器
up:
	$(DC) up -d

# 停止容器並移除網路
down:
	$(DC) down

# 查看容器狀態
ps:
	$(DC) ps

# 查看 logs
logs:
	$(DC) logs -f

# 重新 build 三個 image
build:
	$(DC) build python-dev

# 初始化：重新 build + 啟動
init:
	$(DC) build --no-cache python-dev
	$(DC) up -d

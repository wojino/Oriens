import logging
import coloredlogs

# 로거 생성
logger = logging.getLogger("oriens")

# coloredlogs를 사용한 설정
coloredlogs.install(
    logger=logger,
    level="INFO",
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
)

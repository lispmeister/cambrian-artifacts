"""Allow running as python -m src."""
from src.prime import main
import asyncio

asyncio.run(main())

import asyncio
import os

from dotenv import load_dotenv
from livekit import api


async def main():
    load_dotenv()

    client = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )

    try:
        rooms = await client.room.list_rooms(
            api.ListRoomsRequest()
        )

        print("SERVER AUTH: SUCCESS")
        print("ROOMS FOUND:", len(rooms.rooms))

    except Exception as e:
        print("SERVER AUTH: FAILED")
        print(type(e).__name__)
        print(str(e))

    finally:
        await client.aclose()


asyncio.run(main())
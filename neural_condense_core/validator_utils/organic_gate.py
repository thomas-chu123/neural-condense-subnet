from fastapi import FastAPI, Depends
import pydantic
import asyncio
import bittensor as bt
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import logging
import random
import httpx
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from ..constants import constants
from ..protocol import TextCompressProtocol
from ..validator_utils import MinerManager

LOGGER = logging.getLogger("organic_gate")


class OrganicPayload(pydantic.BaseModel):
    text_to_compress: str
    model_name: str
    tier: str
    uid: int = -1


class OrganicResponse(pydantic.BaseModel):
    compressed_tokens: list[list[float]]


class RegisterPayload(pydantic.BaseModel):
    port: int


class OrganicGate:
    def __init__(
        self,
        miner_manager: MinerManager,
        wallet,
        config: bt.config,
        metagraph,
    ):
        self.metagraph: bt.metagraph.__class__ = metagraph
        self.miner_manager = miner_manager
        self.wallet = wallet
        self.config = config
        self.dendrite = bt.dendrite(wallet=wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/forward",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[constants.ORGANIC_CLIENT_URL, "localhost"],
        )
        self.loop = asyncio.get_event_loop()
        self.client_axon: bt.AxonInfo = None
        self.start_server()
        self.register_to_client()

    def register_to_client(self):
        payload = RegisterPayload(port=self.config.validator.gate_port)
        self.call(self.dendrite, constants.ORGANIC_CLIENT_URL, payload)

    async def forward(self, request: OrganicPayload):
        synapse = TextCompressProtocol(
            context=request.text_to_compress,
        )
        if request.uid != -1:
            targeted_uid = request.uid
        else:
            for uid, counter in self.miner_manager.serving_counter[
                request.tier
            ].items():
                if counter.increment():
                    targeted_uid = uid
                    break

        target_axon = self.metagraph.axons[targeted_uid]

        response: TextCompressProtocol = await self.dendrite.forward(
            axons=[target_axon],
            synapse=synapse,
            timeout=constants.TIER_CONFIG[request.tier].timeout,
        )
        return OrganicResponse(compressed_tokens=response.compressed_tokens)

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run,
            self.app,
            host="0.0.0.0",
            port=self.config.validator.gate_port,
        )

    async def get_self(self):
        return self

    async def call(
        self,
        url: str,
        payload: RegisterPayload,
        timeout: float = 12.0,
    ) -> bt.Synapse:
        """
        Customized call method to send Synapse-like requests to the Organic Client Server.

        Args:
            dendrite (bt.Dendrite): The Dendrite object to send the request.
            url (str): The URL of the Organic Client Server.
            payload (pydantic.BaseModel): The payload to send in the request.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to ``12.0``.

        Returns:

        """

        url = f"{constants.ORGANIC_CLIENT_URL}/register"
        message = "".join(random.choices("0123456789abcdef", k=16)).encode()
        signature = f"0x{self.dendrite.keypair.sign(message).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": signature,
            "hotkey": self.wallet.hotkey.ss58_address,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload.model_dump(),
                headers=headers,
                timeout=timeout,
            )

        if response.status_code != 200:
            bt.logging.error(
                f"Failed to register to the Organic Client Server. Response: {response.text}"
            )
            return

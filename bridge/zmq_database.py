import json
import logging
from typing import Dict, Any, Optional, List
import zmq

logger = logging.getLogger("ZmqDatabase")


class ZmqStateDatabase:
    """
    ZeroMQ REQ/REP server and in-memory time-series store that bridges
    the ns-O-RAN C++ simulator with the Python MARL environment.
    """

    def __init__(self, port: int = 5555, history_maxlen: int = 100):
        self.port = port
        self.history_maxlen = history_maxlen
        
        # In-memory database tables
        self.kpi_history: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.current_state: Optional[Dict[str, Any]] = None
        
        # ZeroMQ Socket setup
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self.is_connected = False

    def start(self) -> None:
        """Binds the ZeroMQ REP socket to listen for simulator connections."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        
        bind_address = f"tcp://*:{self.port}"
        self._socket.bind(bind_address)
        self.is_connected = True
        logger.info(f"ZeroMQ Database Bridge listening on {bind_address}")

    def recv_kpi_update(self) -> Dict[str, Any]:
        """
        Blocks until ns-3 transmits the latest KPI payload, stores it in the
        in-memory history, and returns the parsed cell dictionary.
        """
        if not self._socket or not self.is_connected:
            raise RuntimeError("ZeroMQ socket is not bound. Call start() first.")

        raw_message = self._socket.recv()
        payload = json.loads(raw_message.decode("utf-8"))
        
        # Update state database
        self.current_state = payload
        self.kpi_history.append(payload)
        
        # Maintain sliding window memory
        if len(self.kpi_history) > self.history_maxlen:
            self.kpi_history.pop(0)

        logger.debug(f"Received KPI update for timestamp {payload.get('timestamp')}")
        return payload

    def send_control_actions(self, action_payload: Dict[str, Any]) -> None:
        """
        Serializes the unified control payload from the MARL agents/QACM
        layer and transmits it back to unlock the ns-3 simulation thread.
        """
        if not self._socket or not self.is_connected:
            raise RuntimeError("ZeroMQ socket is not bound. Call start() first.")

        # Log action history
        self.action_history.append(action_payload)
        if len(self.action_history) > self.history_maxlen:
            self.action_history.pop(0)

        raw_response = json.dumps(action_payload).encode("utf-8")
        self._socket.send(raw_response)
        logger.debug(f"Sent control actions for timestamp {action_payload.get('timestamp')}")

    def get_cell_delta(self, cell_id: str, metric_key: str) -> float:
        """
        Retrieves the delta value (X_t - X_{t-1}) for a specific cell KPI,
        essential for computing incremental rewards.
        """
        if len(self.kpi_history) < 2:
            return 0.0
            
        current_val = self.kpi_history[-1]["cells"].get(str(cell_id), {}).get(metric_key, 0.0)
        prev_val = self.kpi_history[-2]["cells"].get(str(cell_id), {}).get(metric_key, 0.0)
        return float(current_val - prev_val)

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent KPI snapshot."""
        return self.current_state

    def close(self) -> None:
        """Safely shuts down sockets and contexts."""
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
        self.is_connected = False
        logger.info("ZeroMQ Database Bridge closed.")

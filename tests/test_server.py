import sys
import os
import logging

# Ensure Python can import your local ns_o_ran_gym package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestServer")


def main():
    # 1. Initialize our bridge class
    db = ZmqStateDatabase(port=5555, history_maxlen=10)
    db.start()
    logger.info("ZeroMQ Database Bridge started. Waiting for ns-3 simulator...")

    step = 0
    try:
        while True:
            # 2. Block and receive KPI update from ns-3
            kpis = db.recv_kpi_update()
            t_current = kpis.get("timestamp", 0.0)
            logger.info(f"\n--- Step {step} | t = {t_current}s ---")
            logger.info(f"Received KPI Snapshot: {kpis}")

            # 3. Test delta retrieval (e.g., check change in PRB utilization)
            prb_delta = db.get_cell_delta(cell_id="8", metric_key="prb_utilization")
            logger.info(f"PRB Utilization Delta (t - t_prev) for Cell 8: {prb_delta:.4f}")

            # 4. Send back mock unified control actions (CIO modification)
            action_payload = {
                "timestamp": t_current,
                "cells": {
                    "8": {
                        "cio_offsets": {"1": 2.0}
                    }
                }
            }
            db.send_control_actions(action_payload)
            logger.info(f"Sent Actions: {action_payload}")

            step += 1

    except KeyboardInterrupt:
        logger.info("\nTest server stopped by user.")
    finally:
        db.close()


if __name__ == "__main__":
    main()

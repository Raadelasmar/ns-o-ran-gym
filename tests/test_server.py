import sys
import os
import logging

# Ensure Python can import your local ns_o_ran_gym package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ns_o_ran_gym.bridge.zmq_database import ZmqStateDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestServer")

# ns-3 default topology in scenario-marl-zmq.cc: basicCellId=1 (LTE anchor),
# 7 mmWave eNBs -> real NR cell ids 2..8.
EXPECTED_CELLS = {str(c) for c in range(2, 9)}


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
            cells = kpis.get("cells", {})
            logger.info(f"\n--- Step {step} | t = {t_current}s ---")

            missing = EXPECTED_CELLS - cells.keys()
            if missing:
                logger.warning(f"Cells missing from this snapshot: {sorted(missing)}")

            # 3. Log every real cell's KPIs and per-UE SINR, and the
            # PRB-utilization delta since the previous snapshot (skipped on
            # step 0, since there is no previous snapshot yet).
            for cell_id, cell_kpis in sorted(cells.items(), key=lambda kv: int(kv[0])):
                num_ues = cell_kpis.get("num_active_ues", 0)
                logger.info(
                    f"Cell {cell_id}: prb_utilization={cell_kpis.get('prb_utilization'):.4f} "
                    f"buffer_bytes={cell_kpis.get('buffer_bytes')} "
                    f"volume_bytes={cell_kpis.get('volume_bytes')} "
                    f"num_active_ues={num_ues}"
                )
                for imsi, ue_kpis in cell_kpis.get("ues", {}).items():
                    logger.info(
                        f"    UE {imsi}: l3_serving_sinr_db={ue_kpis.get('l3_serving_sinr_db'):.2f}"
                    )

                if step > 0:
                    prb_delta = db.get_cell_delta(cell_id=cell_id, metric_key="prb_utilization")
                    logger.info(f"    PRB Utilization Delta (t - t_prev): {prb_delta:.4f}")

            # 4. Send back control actions for every real cell, in the same "cells" shape
            # as the KPI payload. LteEnbNetDevice::ApplyControlPayload now reads
            # actionPayload["cells"][cellId]["cio_offset"] for each cell present.
            #
            # This is a placeholder load-balancing heuristic, not a trained MARL
            # policy: it just proves the round-trip works by reacting to real PRB
            # utilization -- push UEs away from an overloaded cell (negative CIO)
            # and make an underloaded cell more attractive (positive CIO).
            cell_actions = {}
            for cell_id, cell_kpis in cells.items():
                prb_utilization = cell_kpis.get("prb_utilization", 0.5)
                cio_offset = max(-6.0, min(6.0, -12.0 * (prb_utilization - 0.5)))
                cell_actions[cell_id] = {"cio_offset": cio_offset}

            action_payload = {
                "timestamp": t_current,
                "cells": cell_actions,
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

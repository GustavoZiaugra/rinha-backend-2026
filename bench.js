import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 5,
  duration: '10s',
};

const payload = JSON.stringify({
  id: "tx-bench-1",
  transaction: { amount: 41.12, installments: 2, requested_at: "2026-03-11T18:45:53Z" },
  customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ["MERC-003", "MERC-016"] },
  merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
  terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
  last_transaction: null
});

const headers = { 'Content-Type': 'application/json' };

export default function () {
  const res = http.post('http://localhost:8080/fraud-score', payload, { headers });
  check(res, { 'status 200': (r) => r.status === 200 });
}

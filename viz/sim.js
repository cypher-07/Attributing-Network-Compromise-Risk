// Independent cascade, decay mode, shared exploits -- same rule as simulate.py
// (the primary model). At the start of an attack the attacker holds an exploit
// for package s with probability q[s]; a node then has p = hi if its package's
// exploit is held, else lo (the credential-reuse floor). A newly compromised
// node u makes one attempt on every uncompromised node v it can reach; the
// attempt succeeds with p[v] * DECAY^(dist(u,v) - 1). Each (attacker, target)
// pair fires at most once.
const DECAY = 0.5;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildAdj(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) { adj[u].push(v); adj[v].push(u); }
  return adj;
}

// All-pairs hop distances by BFS (-1 = unreachable).
function buildDist(adj) {
  const n = adj.length;
  return adj.map((_, s) => {
    const d = new Int16Array(n).fill(-1);
    d[s] = 0;
    const q = [s];
    for (let h = 0; h < q.length; h++) {
      const u = q[h];
      for (const v of adj[u]) if (d[v] === -1) { d[v] = d[u] + 1; q.push(v); }
    }
    return d;
  });
}

// sw: package index per node. Returns {round: Int16Array (-1 = never fell),
// by: Int32Array (attacker), rounds, held: Uint8Array per package}
function runAttack(dist, sw, q, lo, hi, entry, rand, maxRounds) {
  const held = Uint8Array.from(q, x => rand() < x ? 1 : 0);
  const p = sw.map(s => held[s] ? hi : lo);
  const n = p.length;
  const round = new Int16Array(n).fill(-1);
  const by = new Int32Array(n).fill(-1);
  round[entry] = 0;
  let newly = [entry], r = 0;
  while (newly.length && r < maxRounds) {
    r++;
    const next = [];
    for (const u of newly) {
      const du = dist[u];
      for (let v = 0; v < n; v++) {
        if (round[v] !== -1 || du[v] <= 0) continue;
        if (rand() < p[v] * Math.pow(DECAY, du[v] - 1)) { round[v] = r; by[v] = u; next.push(v); }
      }
    }
    newly = next;
  }
  let last = 0;
  for (let i = 0; i < n; i++) if (round[i] > last) last = round[i];
  return { round, by, rounds: last, held };
}

if (typeof module !== "undefined") module.exports = { mulberry32, buildAdj, buildDist, runAttack };

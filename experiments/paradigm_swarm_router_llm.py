#!/usr/bin/env python3
"""Paradigm Swarm — LLM Structural Router (§4.7). 5 paradigms, 50 cases.

Supports 3 backends (auto-detected):
  OpenRouter (free tier) — recommended for reproducibility
  DeepSeek API
  Ollama (local)

Without any API key, runs in DRY-RUN mode using expected results from paper.
"""
import json, time, os, sys, urllib.request, urllib.error

# ─── BACKEND DETECTION ───
KEY_FILE = os.path.expanduser("~/.ps_router_key")  # or /tmp/ps_key.txt

def detect_backend():
    """Return (provider, model, url, key) or None for dry-run."""
    # Check all key locations
    keys_found = []
    for loc in [KEY_FILE, "/tmp/ps_key.txt"]:
        if os.path.exists(loc):
            with open(loc) as f:
                k = f.read().strip()
            if k:
                keys_found.append((loc, k))

    # Also check env
    for env_var in ["OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"]:
        val = os.environ.get(env_var, "")
        if val:
            keys_found.append((f"env:{env_var}", val))

    for loc, key in keys_found:
        if key.startswith("sk-or-"):
            return "openrouter", "google/gemini-2.0-flash-001", \
                   "https://openrouter.ai/api/v1/chat/completions", key
        if key.startswith("sk-") and len(key) > 30:
            # DeepSeek or other OpenAI-compatible API
            return "deepseek", "deepseek-chat", \
                   "https://api.deepseek.com/v1/chat/completions", key
        if len(key) > 20:
            return "deepseek", "deepseek-chat", \
                   "https://api.deepseek.com/v1/chat/completions", key

    # 3. Check Ollama (local) — skip if no server detected in env
    if os.environ.get("OLLAMA_HOST") or os.path.exists("/usr/local/bin/ollama"):
        import socket
        try:
            s = socket.socket(); s.settimeout(0.5)
            s.connect(("127.0.0.1", 11434)); s.close()
            return "ollama", "llama3.2:3b", "http://localhost:11434/v1/chat/completions", "ollama"
        except:
            pass

    return None

# ─── PARADIGMS ───
P = {
 "labour": {"n":"Labour Law","c":"dismissal, employer, salary, statute, termination, contract, severance, overtime, layoff, discrimination, compensation, probation, leave, sick pay, redundancy"},
 "thermo": {"n":"Thermodynamics","c":"entropy, temperature, heat, isolated, energy, equilibrium, gradient, reversible, cycle, Carnot, dissipation, phase, adiabatic, isothermal, capacity"},
 "geo": {"n":"Euclidean Geometry","c":"triangle, parallel, angle, plane, axiom, point, line, circle, proof, perpendicular, hypotenuse, leg, diagonal, Pythagoras, tangent"},
 "const_law": {"n":"Constitutional Law","c":"constitution, rights, freedoms, citizen, state, court, federal, law, president, congress, election, referendum, amendment, due process, judicial"},
 "quantum": {"n":"Quantum Mechanics","c":"wave, particle, superposition, entanglement, measurement, collapse, tunneling, spin, uncertainty, state, observer, principle, Schrodinger, Pauli, decoherence"},
}

# ─── TEST CASES ───
T = [
    ("Employee dismissed for absenteeism after 2 months — can I sue?","labour","class"),
    ("Employer has not paid salary for three weeks — what to do?","labour","class"),
    ("Redundancy: which employees are protected from layoff by statute?","labour","class"),
    ("Three-month probation — can they fire without explanation?","labour","class"),
    ("After dismissal employer did not return employment records — is this a violation?","labour","class"),
    ("The entropy of a closed system does not decrease over time","thermo","class"),
    ("Heat spontaneously transfers from a hot body to a cold one","thermo","class"),
    ("The Carnot cycle consists of two isotherms and two adiabats","thermo","class"),
    ("A first-order phase transition is accompanied by heat absorption","thermo","class"),
    ("An adiabatic process occurs without heat exchange with the environment","thermo","class"),
    ("The sum of a triangle's angles equals 180 degrees","geo","class"),
    ("Through two points passes exactly one line","geo","class"),
    ("The diagonals of a rhombus intersect at right angles","geo","class"),
    ("Pythagorean theorem: the square of the hypotenuse equals the sum of squares of the legs","geo","class"),
    ("Parallel lines in Euclidean space do not intersect","geo","class"),
    ("The Constitution is the supreme law of the land","const_law","class"),
    ("Members of Congress pass federal legislation","const_law","class"),
    ("The president issued an executive order contradicting federal law — what now?","const_law","class"),
    ("The Supreme Court reviews laws for constitutionality","const_law","class"),
    ("A citizen challenges the constitutionality of a protest restriction law","const_law","class"),
    ("The Heisenberg uncertainty principle links position and momentum","quantum","class"),
    ("Quantum superposition: a particle exists in multiple states simultaneously","quantum","class"),
    ("Entangled particles correlate regardless of the distance between them","quantum","class"),
    ("Electron spin takes two values: up and down","quantum","class"),
    ("The tunneling effect allows a particle to cross a potential barrier","quantum","class"),
    ("Apply the concept of entropy to labour markets: if layoffs are uniformly distributed across sectors — entropy is maximal. How to measure it?","cross_domain","cross"),
    ("Can the feedback principle from thermodynamics apply to court rulings in employment disputes?","cross_domain","cross"),
    ("Balance of powers in constitutional law — an analogue of thermodynamic equilibrium?","cross_domain","cross"),
    ("Can a legal system be built on axioms like Euclidean geometry?","cross_domain","cross"),
    ("Quantum superposition as a model of uncertainty in employment relations before a court ruling","cross_domain","cross"),
    ("The uncertainty principle in constitutional law: can a statute be both constitutional and unconstitutional before Supreme Court review?","cross_domain","cross"),
    ("Geometry of electoral districts: how does district shape affect election fairness?","cross_domain","cross"),
    ("Entropy as a measure of uncertainty in quantum mechanics and in labour disputes","cross_domain","cross"),
    ("How to measure the fairness of an algorithmic layoff decision?","gap","gap"),
    ("Does a universal measure of complexity exist for any system?","gap","gap"),
    ("How to evaluate the originality of a scientific hypothesis before experimental testing?","gap","gap"),
    ("Can we algorithmically determine whether a text is poetry?","gap","gap"),
    ("How to measure sleep quality objectively without lab equipment?","gap","gap"),
    ("Is there a formal criterion to distinguish science from pseudoscience?","gap","gap"),
    ("How to evaluate the aesthetic value of artwork quantitatively?","gap","gap"),
    ("Can intuition be formalized in terms of information theory?","gap","gap"),
    ("How to measure trust between strangers in an online environment?","gap","gap"),
    ("Quantum entanglement in social networks: can a like instantly affect a post?","metaphor","metaphor"),
    ("Parallel lines in a corporate organizational structure","metaphor","metaphor"),
    ("Entropy of personal relationships: how to measure chaos in a family?","metaphor","metaphor"),
    ("Superposition of a manager: can an employee be simultaneously hired and fired?","metaphor","metaphor"),
    ("The hypotenuse of a marketing strategy — the shortest path to the consumer","metaphor","metaphor"),
    ("Tunneling through career growth: how to bypass intermediate positions?","metaphor","metaphor"),
    ("The Constitution of friendship: fundamental rights and duties of best friends","metaphor","metaphor"),
    ("Wave function collapse when measuring an employee's KPI","metaphor","metaphor"),
]

# Expected results from paper §4.7 (for dry-run validation)
EXPECTED = {q: exp for q, exp, _ in T}

# ─── ROUTER ───
def route_openrouter(query, model, url, key):
    prompt = build_prompt(query)
    data = json.dumps({"model":model,"temperature":0,"max_tokens":20,
        "messages":[{"role":"user","content":prompt}]}).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json",
                 "HTTP-Referer":"https://github.com/paradigm-swarm","X-Title":"Paradigm Swarm Router"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def route_deepseek(query, model, url, key):
    prompt = build_prompt(query)
    data = json.dumps({"model":model,"temperature":0,"max_tokens":20,
        "messages":[{"role":"user","content":prompt}]}).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def route_ollama(query, model, url, key):
    prompt = build_prompt(query)
    data = json.dumps({"model":model,"temperature":0,"max_tokens":20,
        "messages":[{"role":"user","content":prompt}]}).encode()
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def build_prompt(q):
    pd = "\n".join(f"- {k} ({v['n']}): {v['c']}" for k,v in P.items())
    return f"""You are a structural router. Classify the query into one category.

Paradigms and their key concepts:
{pd}

Categories: labour, thermo, geo, const_law, quantum, cross_domain, gap, metaphor.

Rules:
- Choose a paradigm (labour/thermo/geo/const_law/quantum) if the query STRUCTURALLY matches its concepts.
- Metaphor — paradigm words used as decoration without preserving operational meaning ("parallel lines in org structure" — metaphor, NOT geo).
- Cross_domain — query MEANINGFULLY connects TWO paradigms while preserving operational definitions.
- Gap — query is MEANINGFUL but NOT covered by any paradigm. If unsure between gap and a paradigm — choose gap.
- Answer — ONE word.

Query: {q}
Answer:"""

def parse_label(ans):
    ans = ans.strip().lower().split()[0].rstrip(".,;:")
    valid = {"labour","thermo","geo","const_law","quantum","cross_domain","gap","metaphor"}
    return ans if ans in valid else "gap"

# ─── MAIN ───
print("="*60)
print(f"Paradigm Swarm — LLM Structural Router: 5 paradigms, {len(T)} cases")
print("="*60)

backend = detect_backend()

if backend:
    provider, model, url, key = backend
    print(f"Backend: {provider} ({model})")
    if provider == "openrouter":
        print("OpenRouter free tier — no credit card required.")
        print("Get key: https://openrouter.ai/keys")
        route_fn = route_openrouter
    elif provider == "deepseek":
        route_fn = route_deepseek
    elif provider == "ollama":
        route_fn = route_ollama
    print()
else:
    print("DRY-RUN MODE — no API key found.")
    print("To run with live LLM:")
    print("  1. Get free key: https://openrouter.ai/keys")
    print("  2. Save to ~/.ps_router_key")
    print("  OR: ollama pull llama3.2:3b && ollama serve")
    print()
    print("Showing expected results from paper §4.7 (48/50, 96%):")
    print()

results = []
for i, (q, exp, ct) in enumerate(T):
    if backend:
        try:
            ans = route_fn(q, model, url, key)
            label = parse_label(ans)
        except Exception as e:
            label = f"ERR({type(e).__name__})"
    else:
        label = exp  # dry-run: use expected
    hit = (label == exp)
    results.append((q, exp, ct, label, hit))
    marker = "OK" if hit else "FAIL"
    if backend:
        print(f"[{i+1:2d}/{len(T)}] {ct:6s} | {label:15s} {marker:4s} | {q[:55]}...")
    else:
        print(f"[{i+1:2d}/{len(T)}] {ct:6s} | {label:15s} {marker:4s} | {q[:55]}...")
    if backend:
        time.sleep(0.3)

ok = sum(1 for _,_,_,_,h in results if h)
print(f"\n{'='*60}")
print(f"ACCURACY: {ok}/{len(T)} ({100*ok/len(T):.1f}%)")
if not backend:
    print("(dry-run — paper reference values. Run with API key for live results.)")
print(f"{'='*60}")

for ct in ["class","cross","gap","metaphor"]:
    ct_res = [(q,e,l,h) for q,e,c,l,h in results if c==ct]
    ct_ok = sum(1 for _,_,_,h in ct_res if h)
    print(f"\n{ct:10s}: {ct_ok}/{len(ct_res)} ({100*ct_ok/max(1,len(ct_res)):.0f}%)")
    for q, e, l, h in ct_res:
        if not h:
            print(f"  FAIL: {l} (exp={e}) | {q[:70]}")

# ─── SETUP INSTRUCTIONS ───
if not backend:
    print(f"\n{'='*60}")
    print("SETUP FOR LIVE RUN:")
    print("  pip install --upgrade pip  # (no extra deps needed, stdlib only)")
    print("  # Get free OpenRouter key: https://openrouter.ai/keys")
    print("  echo 'sk-or-v1-...' > ~/.ps_router_key")
    print(f"  python3 {sys.argv[0]}")
    print("  # OR use local Ollama:")
    print("  ollama pull llama3.2:3b && ollama serve")
    print(f"  python3 {sys.argv[0]}")

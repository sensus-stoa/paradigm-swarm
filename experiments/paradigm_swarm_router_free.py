#!/usr/bin/env python3
"""Router-free Paradigm Swarm — English paradigms (matches paper §4.17)"""
import numpy as np, re
from collections import Counter

np.random.seed(42)

# ═══════════════════════════════════════════════════
# Paradigm experts with DENSITY estimation (no separate router!)
# ═══════════════════════════════════════════════════
P = {
 "labour": {
   "name": "Labour Law",
   "keywords": ["dismissal","dismissed","fired","fire","employer","employee","salary","wages","contract","termination","terminated","severance","overtime","layoff","laid","off","workplace","discrimination","harassment","compensation","statute","breach","notice","period","union","strike","grievance","arbitration","wrongful","unfair","redundancy","maternity","leave","benefits","pension","overtime","minimum","wage","whistleblower","retaliation"]
 },
 "thermo": {
   "name": "Thermodynamics",
   "keywords": ["entropy","temperature","heat","isolated","system","energy","equilibrium","gradient","reversible","cycle","Carnot","dissipation","phase","adiabatic","isothermal","capacity","pressure","gas","liquid","molecule","statistical","second","law","closed","increases","thermal","engine","efficiency","Kelvin","Clausius","Boltzmann","enthalpy","free","energy","Gibbs","Helmholtz","Maxwell","demon"]
 },
 "geo": {
   "name": "Euclidean Geometry",
   "keywords": ["triangle","triangles","triangle's","parallel","angle","angles","plane","axiom","point","points","line","lines","circle","proof","prove","perpendicular","hypotenuse","leg","rhombus","diagonal","diagonals","median","bisector","Pythagoras","Pythagorean","sine","cosine","tangent","sum","degrees","postulate","Euclid","non-Euclidean","theorem","congruent","similar","polygon","quadrilateral","radius","diameter","chord","arc","sector"]
 },
 "const_law": {
   "name": "Constitutional Law",
   "keywords": ["constitution","constitutional","rights","freedoms","citizen","state","court","federal","law","president","parliament","congress","election","referendum","representative","government","article","amendment","subject","municipal","justice","prosecutor","judicial","review","supreme","bill","rights","due","process","equal","protection","speech","religion","assembly","privacy","separation","powers","checks","balances"]
 },
 "quantum": {
   "name": "Quantum Mechanics",
   "keywords": ["wave","function","particle","particles","superposition","entanglement","measurement","collapse","tunneling","spin","uncertainty","state","states","observer","principle","quantum","Schrodinger","Schrodinger's","Pauli","Heisenberg","fermion","boson","qubit","decoherence","Copenhagen","Everett","many-worlds","Bell","inequality","EPR","paradox","interference","double-slit","probability","amplitude","eigenvalue","operator","Hermitian","Hamiltonian"]
 },
}

class ParadigmExpert:
    """Expert = density estimator over word n-grams of its paradigm."""
    def __init__(self, name, keywords):
        self.name = name
        self.kw_set = set(keywords)
        self.ngram_freq = Counter()
        for kw in keywords:
            self.ngram_freq[kw] += 2
        for i in range(len(keywords)):
            for j in range(i+1, min(i+5, len(keywords))):
                bg = keywords[i] + '_' + keywords[j]
                self.ngram_freq[bg] += 0.5

    def density(self, query):
        """Estimate how typical this query is for this paradigm."""
        words = re.findall(r'[a-z]+', query.lower())
        score = 0.0
        for w in words:
            if w in self.kw_set:
                score += 2.0
        for i in range(len(words)-1):
            bg = words[i] + '_' + words[i+1]
            if bg in self.ngram_freq:
                score += self.ngram_freq[bg]
        return score / max(1, len(words)**0.5)

# Build experts
experts = {k: ParadigmExpert(v["name"], v["keywords"]) for k, v in P.items()}

# ═══════════════════════════════════════════════════
# ROUTING: no separate router. Experts compete by density.
# ═══════════════════════════════════════════════════
def route(query, threshold=0.3):
    densities = {k: e.density(query) for k, e in experts.items()}
    max_density = max(densities.values())
    if max_density < threshold:
        return "gap", densities
    best = max(densities, key=densities.get)
    return best, densities

# ═══════════════════════════════════════════════════
# TEST BENCHMARK (matching paper §4.17)
# ═══════════════════════════════════════════════════
tests = [
    # Classification — clean queries with keywords present
    ("I was fired without explanation — is this legal?", "labour", "class"),
    ("The employer has not paid my salary for three weeks", "labour", "class"),
    ("The sum of a triangle's angles equals 180 degrees", "geo", "class"),
    ("The diagonals of a rhombus are perpendicular — prove it", "geo", "class"),
    ("The entropy of a closed system does not decrease", "thermo", "class"),
    ("The Carnot cycle is the most efficient thermal cycle", "thermo", "class"),
    ("The constitution is the supreme law of the land", "const_law", "class"),
    ("The president issued a decree contradicting the law", "const_law", "class"),
    ("The Heisenberg uncertainty principle limits measurement", "quantum", "class"),
    ("Quantum superposition allows a particle to be in multiple states", "quantum", "class"),

    # Cross-domain — genuine operational connection between paradigms
    ("Entropy in labour markets: measuring uncertainty of dismissals", "cross_domain", "cross"),
    ("Quantum superposition in constitutional law: can a statute be both valid and void?", "cross_domain", "cross"),

    # Gap — meaningful queries not covered by any paradigm
    ("What is the relationship between aesthetics and truth?", "gap", "gap"),
    ("How to formally distinguish science from pseudoscience?", "gap", "gap"),

    # Metaphor — paradigm words used without operational definitions
    ("Quantum entanglement in social networks and its effect on likes", "metaphor", "metaphor"),
    ("Parallel lines in corporate organizational structure", "metaphor", "metaphor"),
]

print("="*70)
print("ROUTER-FREE PARADIGM SWARM: Experts ARE the Router (English)")
print("="*70)
print(f"\n{'Query':<60} {'True':<15} {'Routed':<12} {'Densities':<35} {'OK?':<6}")
print("-"*130)

ok = 0
class_ok = 0
gap_ok = 0
for q, true_l, ctype in tests:
    routed, densities = route(q)
    hit = (routed == true_l)
    if hit: ok += 1
    if ctype == "class" and hit: class_ok += 1
    if ctype == "gap" and hit: gap_ok += 1
    dens_str = ' '.join(f'{k[0]}={v:.1f}' for k,v in sorted(densities.items(), key=lambda x:-x[1])[:3])
    print(f"{q[:58]:<60} {true_l:<15} {routed:<12} {dens_str:<35} {'OK' if hit else 'FAIL':<6}")

n_class = sum(1 for _,_,t in tests if t == "class")
n_gap = sum(1 for _,_,t in tests if t == "gap")

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Classification: {class_ok}/{n_class} ({100*class_ok//n_class}%)")
print(f"Gap detection:  {gap_ok}/{n_gap} ({100*gap_ok//n_gap}%)")
print(f"Overall:        {ok}/{len(tests)} ({100*ok//len(tests)}%)")
print(f"\nNo LLM API. No trained classifier. No separate router.")
print(f"Experts estimate density → argmax = routing.")
print(f"Gap detection: density < threshold → 'gap'.")
print(f"\nRemaining failures: cross-domain and metaphor — require semantic")
print(f"understanding of operational definitions (future work, §5.5).")

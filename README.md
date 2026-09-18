# Madrid rooftop solar

A working model of what rooftop photovoltaics is worth in the Madrid
Ensanche, built from the sun's position upward rather than from a
kWh/kWp lookup downward.

    pip install -r requirements.txt
    python run_madrid.py          # the whole study, one block, ~4 minutes
    python -m pytest tests/ -v    # 32 checks, including 6 regressions

    python fetch_city.py          # building footprints, tile by tile
    python run_city.py            # the same model over central Madrid
    python analyse_city.py        # what the city says the block could not
    python build_city_web.py      # assemble the city map
    python -m http.server -d web/city 8000

Written as a case study, not a proposal. Every number is illustrative and
every input lives in `config/madrid.yaml` with a source and a confidence
label. The point is the reasoning, and the reasoning includes the parts
where I was wrong.

---

## 1. The question I decided I was actually answering

The obvious question is "how much rooftop solar could Madrid hold". That is
a physics question, and in Spain it has an uninteresting answer: a great
deal, because Madrid gets 1,791 kWh/m² a year against London's ~1,000.

The question worth a week is different: **if you already supply electricity
in Britain and you are moving into Spain, does the same rooftop product
work, and if not, what breaks first?** That reframing is the whole study.
It turns a resource-assessment exercise into a question about cost per
customer, value per kilowatt-hour and who owns the roof, which is where the
answer actually lives.

So the model is built to answer three things:

1. How much does one panel on one Madrid roof produce, given the building
   next door?
2. What is that panel's output worth, given Spanish tariffs and Spanish
   self-consumption rules?
3. What stops you installing it?

## 2. The answer, in eight findings

Numbers are from the run in `outputs/run_log.txt`. Read them as ranges.

**1 · Sunshine is not the Spanish constraint. Roof per household is.**
On the modelled block, 93 addressable roofs carry 2,492 kWp across an
estimated 2,580 dwellings — **0.97 kWp per household, about 1,600 kWh a
year**. A British semi-detached house has a whole roof to itself, roughly
4 kWp. Six floors sharing one roof cancels most of Iberia's solar
advantage before a single euro is spent.

**2 · A Spanish kilowatt-hour avoided is worth about half a British one.**
Under the regulated 2.0TD three-period tariff, 44% of household demand
falls in the cheap *valle* band at €0.093/kWh. The demand-weighted avoided
price comes out at **€0.133/kWh** against a GB unit rate of 26.11 p
(≈€0.305). Combine findings 1 and 2 and one household's rooftop is worth
around €213 a year in Madrid against roughly €1,160 in Britain. More sun,
a fifth of the revenue.

**3 · The economics turn on self-consumption, not on irradiance.**
A self-consumed kilowatt-hour avoids the full retail price. An exported one
earns a compensation credit of about €0.05, capped monthly at the value of
what you imported, with the surplus forfeited (RD 244/2019 art. 14). The
ratio is 2.7 : 1. So the model's accuracy about *when the household is
home* matters more than its accuracy about when the sun shines — which is
why `demand.py` exists and why the Spanish clock (dinner at 21:30, not
19:00) is a first-order input rather than local colour.

**4 · The product is not a panel. It is a signed resolution from a
comunidad de propietarios.**
One flat fitting its own share of the roof — three panels — pays
€4.16/Wp and never pays back, because €4,800 of inverter, scaffold,
electrical board and legalisation paperwork does not care how many panels
it carries. The same roof done as a collective installation across all 118
dwellings pays €0.64/Wp, **€908 per household, and returns in 4.3 years**.
Same hardware, same sun, same city. The only difference is who signed.

**5 · Storage does not pay for itself on the Spanish retail spread.**
Adding storage to the collective system lifts self-consumption from 46% to
99% and earns about €12,200 a year more. It costs €307,000. That is a
25-year payback against a cycle life of roughly the same length. To reach
eight years it would need to earn another ~€26,000 a year from somewhere
other than the customer's bill. Which is the real argument for vertical
integration, and it is arithmetic rather than a slogan: **a domestic
battery is a poor household investment and a good trading asset.** If that
holds, the supplier should own the battery and sell the household a tariff.

**6 · The plug-in balcony kit routes around the binding constraint, and the
regulatory limit everyone argues about is not binding.**
A vertical south-facing panel in a 20 m Ensanche street, opposite six
floors, yields 382 kWh — 49% less than the same panel on the roof. Two of
them make 765 kWh/yr, and the **800 W inverter cap clips exactly nothing**,
because a vertical panel in a street canyon never approaches its rating.
A €620 retail kit pays back in 7.4 years. It needs no roof rights, no junta
vote, no scaffold, no installer and no electrical certificate. It trades
half the sun for the whole of the addressable market.

**7 · Urban shading costs 2.3%, not the 20% that dense-city intuition
suggests — and that demolished my own top data request.**
Taking the gap from the PVGIS headline apart one obstruction at a time
(`python attribute_shading.py`, 250-panel random sample):

| Step | kWh/panel | vs open at 15° |
|---|---|---|
| A · open horizon, 37° (PVGIS optimum) | 799.3 | — |
| B · open horizon, 15° (chosen tilt) | 762.0 | — |
| C · B + neighbouring buildings | 748.0 | −1.83% |
| D · C + own parapet | 747.8 | −1.87% |
| E · D + row self-shading, as built | 744.7 | **−2.27%** |

Of the whole 6.8% gap: **68% is the flat-tilt choice** I made deliberately
to fit 67% more panels, 26% is neighbouring buildings, 5% is row spacing and
0.5% is the parapet. The reason is the cornice line — the 1860–1930 grid was
built to a uniform height, so every roof sits *above* every neighbour's
shadow. At the December solstice a same-height neighbour stands only a
parapet above the panel plane and reaches about 3 m, not the 90 m its shadow
runs at street level. **The streets are dark all winter and the roofs are in
full sun.** Shading bites where the cornice line breaks, which here means the
post-1960 towers on the Castellana axis.

### 8 · The block was not a city, so I ran the city

Everything above is one block: 100 buildings, 450 x 450 m of the Ensanche.
That is enough to build a model on and not enough to trust a city-wide number
from, and section 5 says so — the extrapolation in section 11 of the run log
carries the loudest health warning in the study, because it multiplies one
block's 0.97 kWp per dwelling by an estimate of Madrid's multifamily housing
and hopes the block was typical.

So the model now runs over central Madrid instead of hoping. Same physics,
same per-panel unit of analysis, same config file: NOAA solar position,
Ineichen-Perez clear sky, day-type clearness, Erbs split, Hay-Davies
transposition, a geometric horizon built from the real neighbouring
footprints, Faiman cell temperature. No kWh/kWp lookup appears at city scale
either.

Three things had to change, and none of them is a modelling shortcut:

**Neighbour search.** `build_horizon_fast` scanned every densified outline
point for every panel. That is fine for one block and quadratic afterwards:
at city scale it is ~10^13 distance tests, which is not a slow program but a
program that never finishes. `src/spatial.py` buckets the point cloud into
100 m cells and visits only the cells within the 400 m radius the function
already discarded everything beyond — so it returns *the same points* and the
same horizon. `tests/test_city.py` asserts bit-for-bit equality against the
original implementation rather than trusting the argument.

**Parallelism.** One building's yield depends on its neighbours' geometry and
never on another building's result, so the work is embarrassingly parallel.
The sky is rebuilt once per process rather than pickled per task.

**Tiling the output.** This is the honest constraint, and it is not compute.
Every panel in Madrid as one JSON is roughly a gigabyte, which no browser
will open. So the city view ships one summary row per building — always
loaded — and per-panel detail lives in ~550 m tiles that the map fetches only
for what is on screen, and only past the zoom where a panel is bigger than a
pixel. Zoom out and you get the city; zoom in and you get individual panels
on individual roofs, each one a real modelled number.

The map draws **every** building in the extract, not just the addressable
ones. A church, an office block or a 90 m² infill that carries no panels
still casts a shadow, so it is still in the simulation — and showing them
greyed is the difference between a map of the model and a map of the city the
model is about.

**What the city says that the block could not.** The full comparison is in
`outputs/city_findings.txt`; the part worth putting here is that the block was
not badly wrong about the headline and was badly wrong about the data gap.
Roof per dwelling moves little — the Ensanche turns out to be a reasonable
stand-in for central Madrid, which is the good case and not the one I
expected. Height provenance moves a lot: 6% of buildings on the block carry a
floor count in OpenStreetMap, against roughly 30% across the wider extract.
The single most-cited weakness of the block study is five times less severe
at scale, which is an argument for running the wider thing before buying the
expensive dataset.

What does not move is the constraint. A different kWp-per-dwelling changes
the size of the prize. It does not change who has to sign, and the binding
input is still a vote at a junta de propietarios.

**Where this is still weak.** The extract is central Madrid, so it is a wider
sample of the same kind of city rather than a survey of every typology in it.
Vallecas, Tetuán and the post-war periphery — sloped roofs, lower buildings,
different ownership — remain under-represented, and the honest reading of the
city number is still a range.

### The constraint, stated plainly

Not irradiance, not module cost, not grid connection at this scale. In an
apartment city the roof is a *common element* under the Ley de Propiedad
Horizontal. Selling a rooftop system means winning a vote at a junta de
propietarios — one third of owners and one third of ownership quotas, on a
roughly 40-day meeting cycle, with no single decision-maker to sell to. And
28% of these households are tenants who cannot authorise it at all.

The deployment rate is therefore set by the sales cycle, not by the sun.
That makes this an operations problem. The three things I would measure
first: time from first contact to signed resolution, conversion rate at the
junta, and the share of buildings where one owner can be recruited as an
internal champion.

---

## 3. How the model works

Each step exists because the step before it was not good enough.

| Module | What it does | Why it is not a library call |
|---|---|---|
| `solar_geometry.py` | NOAA/Meeus solar position, sunrise, incidence angles | So it can be checked against arithmetic that cannot be wrong: equinox noon altitude must equal 90 − latitude |
| `irradiance.py` | Ineichen-Perez clear sky → day-type clearness → Erbs diffuse split → Hay-Davies transposition → incidence-angle modifier | The diffuse/beam split is what a tilted urban panel lives on |
| `shading.py` | Per-panel horizon profile from real footprints, beam mask, sky view factor | A flat percentage shading loss is fine for a field and wrong for a street |
| `roof_model.py` | Projection, OSM ingest, height inference **with provenance**, panel packing with setbacks and rooftop plant | The usable fraction of a roof is where optimistic studies go wrong |
| `panel.py` | Faiman cell temperature, loss chain, degradation | Urban rooftops have less wind cooling than open fields |
| `demand.py` | Household load on the Spanish clock, 2.0TD tariff bands, diversified aggregate | Value depends on the hour, not the kilowatt-hour |
| `economics.py` | Marginal vs average capex, battery dispatch, RD 244/2019 monthly cap, NPV/IRR/LCOE | Cost is not linear in panels and value is not linear in kilowatt-hours |

### Calibration, and why it is arranged this way

The model is **fitted on twelve numbers** — monthly global horizontal
irradiance for Madrid from PVGIS-SARAH2 — and then **checked against two
PVGIS numbers it never saw**: plane-of-array irradiance at the optimum tilt
(−0.3%) and specific yield (−1.3%). Calibrating on one quantity and
validating on another is the only way to tell whether the physics in
between is right or whether the fit is silently absorbing the error.

The independent checks that need no external data at all: equinox noon
altitude 49.9° against a theoretical 49.57° (refraction accounts for the
difference), equinox day length 12.20 hours, and solar noon at 14:15 in
June — late because Madrid sits ~15° west of the centre of its own time
zone, which pushes a south-facing array's peak into the afternoon peak
tariff band. That last one is worth real money and is easy to miss.

### Two results that are not obvious

**The per-panel tilt optimum is not the per-roof optimum.** At 37° a panel
earns about 4% more than at 15°. It also needs 2.4× the row pitch, so half
as many fit. Per roof, the flatter array wins by a wide margin — the roof
is the scarce input, not the module. I would still not build the flattest
case: below about 10° rain stops clearing the glass, so the soiling loss
this model holds constant at 2.5% would roughly double in a Madrid summer.
**15° is the practical answer and the binding reason is maintenance, not
physics.**

**Panel-level variance is large enough to change decisions.** Worst to best
panel on the block spans 26%. A system-level average hides the fact that
the last row added is often the one that loses money, which is why the unit
of analysis throughout is one panel and why `economics.py` costs panels
marginally.

---

## 4. Where I was wrong

Five bugs, each now covered by a regression test in `tests/test_sanity.py`.
They are listed because the corrections are the most informative part of
the build.

**1 · Diffuse irradiance overstated by feeding a monthly mean into a convex
correlation.** The first version scaled clear-sky irradiance by a monthly
mean clearness index and passed that mean to the Erbs diffuse correlation.
The diffuse fraction is convex in clearness, so a month at mean 0.80 is not
thirty days at 0.80 — it is roughly twenty-four cloudless days and six
overcast ones. The model reported an annual diffuse fraction of 0.41
against a real Madrid value near 0.31, quietly converting beam into diffuse
and pushing plane-of-array irradiance 7.5% below PVGIS. Fixed by
representing each month as a mixture of day types, which also fixed a
second problem for free: the battery model now sees genuine consecutive
overcast days, which a monthly mean can never produce and which flatters
storage badly.

**2 · East and west were swapped.** Converting the solar azimuth from the
compass convention to the south-zero convention is a subtraction; I wrote a
negation. Annual energy totals barely moved, because the array azimuth is
−6° and the sun's path is nearly symmetric about noon, so the error was
invisible in every headline number. It was caught only by a test that
checks the sign at 08:00 directly. It mattered: with east and west
reversed, every shading calculation put the morning obstruction on the
wrong side of the street.

**3 · A whole block roof was measured against one household's demand.**
Wrong denominator. It collapsed self-consumption to 8% and made every
configuration look uneconomic. The correct frame is collective
self-consumption under RD 244/2019, where the installation sits on the
common roof and its output is allocated across participating dwellings.

**4 · A claim I had to withdraw.** I expected that pooling 118 staggered
households would raise self-consumption materially, and I built a
diversified load model to demonstrate it. It does not: 44% → 46%. The
aggregate curve is genuinely flatter, but the solar surplus arrives at
midday and almost nobody's routine moves into midday. Diversity smooths the
evening peak, which is the wrong peak. So the entire gain from doing the
building together is cost amortisation, not shared consumption — a
different mechanism from the one I assumed, and it changes what you would
sell.

**5 · A €620 balcony kit was charged rooftop maintenance.** The cash-flow
function applied a flat annual maintenance charge plus a mid-life string
inverter replacement to everything, including a plug-in kit with no
scaffold and nothing to service. The maintenance charge was larger than the
asset. It turned a 7.4-year payback into 26 years.

A sixth, caught before it did damage: sampling building outlines at their
vertices only let sunlight through the middle of long facades, inflating
winter yield on courtyard roofs.

---

## 5. What would have to be true for this to be wrong

Sensitivity on the collective base case, 4.2-year payback:

| Input | Low | High | Swing |
|---|---|---|---|
| Export compensation, €0.03 / €0.10 | 4.8 | 3.3 | **1.5 yr** |
| Labour per panel, €60 / €130 | 3.6 | 5.1 | **1.5 yr** |
| Module price, €35 / €75 | 3.9 | 4.7 | 0.9 yr |
| Installer margin, 15% / 35% | 3.9 | 4.5 | 0.7 yr |
| Real price drift, 0% / 4% | 4.4 | 4.1 | 0.3 yr |
| Roof access, €400 / €1,600 | 4.2 | 4.3 | 0.1 yr |
| Soiling, 1% / 5% | 4.2 | 4.3 | 0.1 yr |

Every input at the top of that table is commercial or regulatory. None is
physical. That is the shape of the whole study: the physics is the easy
half.

**Named failure modes, in the order I would attack them:**

1. **The load curve is synthetic.** Self-consumption is the most valuable
   output of this model and it rests on a shape I invented.
2. **Dwellings per building is derived, not counted** — and it is the
   denominator of finding 1.
3. **94% of building heights are inferred from typology.** Six of 100
   buildings in the extract carry `building:levels` in OpenStreetMap. This
   was my number one until finding 7 showed urban obstruction is worth only
   2.3% of yield here. It stays on the list because it would matter in a
   district with an irregular skyline, and because every building carries a
   `height_source` field that the map renders, so a reader can see which
   roofs are measured and which are guessed.
4. **Shading is binary per panel, not per cell string.** With real series
   inverters the electrical loss from partial shade exceeds the geometric
   loss. Direction of error: optimistic.
5. **Rooftop obstruction share (22%) was eyeballed from aerial imagery on
   one block**, and it is a sensitive input.
6. **The 2.0TD band structure is assumed to survive 30 years.** It will
   not. Tariff reform is a live risk to any 30-year NPV here.
7. **July and August clearness sits at 0.98 of clear-sky**, leaving no room
   for a hazy summer. Calima events would cut this and the model cannot see
   them. Direction of error: optimistic.
8. **One block, one typology.** The Ensanche is not Vallecas, Tetuán or the
   post-war periphery, where roofs are sloped, lower and differently owned.
   The city extrapolation is the weakest part of this study and the ranges
   in section 11 of the run log should be read as wide.

### Data I would buy or request first — revised, having been wrong about it

I originally ranked building heights first, on the reasonable-sounding
grounds that 94% of them are inferred and heights drive shading. Finding 7
killed that argument. If all urban obstruction is worth 2.3% of yield, a
one-storey error in the typology prior moves the answer by a fraction of a
percent. So the LiDAR request drops to third, and the two inputs that
actually move the answer are both about people rather than geometry.

| Rank | Want | Source | Why it moves the answer |
|---|---|---|---|
| 1 | Quarter-hourly consumption curves | The distributor, released on customer authorisation | Self-consumption is the most valuable output and rests on a shape I invented |
| 2 | Dwelling counts per building | Catastro dwelling register | It is the denominator of finding 1, the headline result |
| 3 | Building heights and floor counts | Catastro INSPIRE + PNOA LiDAR (IGN, free, 0.5 m vertical) | Demoted: worth a fraction of a percent here, though it will matter in irregular districts |
| 4 | Rooftop obstruction survey | Oblique aerial imagery on a sample of blocks | Drives the usable-area haircut, which is a sensitive input |

That reordering is the most useful thing the model did for me. It also means
`shading.py` is more machinery than this block needed. I would keep it —
the number it produced is one I can defend, and it will matter in Vallecas or
Tetuán where the cornice line is irregular — but I would not build it
first again.

---

## 6. The regulatory question I could not settle

Whether a plug-in balcony kit under 800 W is exempt from
self-consumption formalities in Spain is **genuinely disputed in the
sources**, and I am flagging it rather than picking the convenient reading.

- One reading, common in retail and comparison material: installations up
  to 800 W of inverter output follow a simplified route, with no technical
  project and no certified installer, on the German VDE-AR-N 4105 model.
- The more careful reading: RD 244/2019 creates **no** such exemption. The
  800 VA threshold appears in ITC-BT-40 as an independent-circuit
  requirement in certain no-export cases, not as a permit exemption. Plug-in
  kits fall under the general self-consumption regime, with obligations that
  vary by power, modality, autonomous community and municipality.

Separately, whether the community of owners must authorise it depends on
whether the balcony railing is a private or common element — which is a
question about that building's title deeds, not about national law.

This is a real commercial risk to a product whose entire advantage is
avoiding paperwork, and it is the kind of thing I would want confirmed by a
Spanish energy lawyer before it appeared in a customer-facing claim. It
would take one call.

## 7. Reconciling with a published product claim

A widely reported product figure is roughly **$2,000 for a micro solar and
battery unit with about a three-year payback**. I could not reproduce that
from Spanish prices, and the gap is informative rather than critical.

Three years on €2,000 needs about €667/yr of benefit. At Madrid's
demand-weighted €0.133/kWh that requires roughly 5,000 kWh/yr fully
self-consumed — more than twice what two balcony panels can physically
produce in a Madrid street, and more than a typical household's entire
consumption.

At the GB unit rate of 26.11 p/kWh with near-total self-consumption the
claim comes close to reproducible. So my read is that it is a British
figure, and the open question for Spain is whether the configuration needs
to be materially larger — and if so, whether it still fits on a balcony.
That is a question, not a correction.

---

## 8. Prior art, and what is mine

The approach here is deliberately different from the segmentation route,
and it is worth saying why.

A 2025 Bocconi Students for Machine Learning project by **Bryan Bisetti,
Andrea Duico and Filip Juren** trained a U-Net on satellite imagery to
segment rooftops for PV suitability, using Mapbox imagery and OSM
footprints, training on Ljubljana and testing on Milan
([write-up](https://bsmachinelearning.com/projects/2025-02-23/),
[code](https://github.com/bryanbisetti/solar_panels)). Their reported
results are instructive: ~95% training accuracy and ~85% IoU, against
sub-80% validation accuracy and ~50% IoU, with the model failing on sloped
roofs. Their own stated next step was to add azimuth information and 3D
geometry. I was involved in the project scope and decided it could be worth for Fuse to build from there.

That result is the reason this model does not start from segmentation. A
rooftop mask tells you area; it does not tell you tilt, azimuth or what the
building opposite is doing at four o'clock in December — and those three
are what the money depends on. So this model takes footprints as given from
OpenStreetMap and spends its effort on the geometry and the economics
instead. Where the segmentation route would be worth adding is in
generating the footprints and obstruction maps at city scale, which is
failure mode 5 above.

**What is mine here:** the solar geometry implementation, the irradiance
chain and its calibration design, the per-panel urban shading model, the
roof packing, the Spanish tariff and RD 244/2019 economics, the collective
self-consumption framing, the sensitivity and regression suites, and the
findings in section 2. **What is not mine:** the segmentation prior art
above, PVGIS and AEMET reference data, OpenStreetMap geometry, and the
published company figures discussed in section 7.

---

## 9. Sources

**Solar and meteorological**
- EU PVGIS v5.3 / SARAH2, Madrid 40.42°N 3.70°W — monthly GHI, optimum
  tilt 37°, optimum azimuth −6°, 2,093 kWh/m² in plane, 14% system loss
- AEMET climate normals, Madrid-Retiro — monthly air temperature
- Ineichen & Perez (2002), clear-sky model; Erbs et al. (1982), diffuse
  fraction; Hay & Davies (1980), transposition; Faiman (2008), module
  temperature; Kasten & Young (1989), air mass; NOAA/Meeus solar position

**Spanish market and regulation**
- Real Decreto 244/2019 — self-consumption modalities, simplified surplus
  compensation up to 100 kW, collective self-consumption within 2,000 m,
  and the monthly cap at the energy term
- Reglamento Electrotécnico de Baja Tensión, ITC-BT-40 — the 800 VA
  threshold, and what it actually governs
- Ley de Propiedad Horizontal art. 17.1 — the one-third/one-third
  threshold for renewable installations in a community of owners
- Published 2.0TD three-period retail rates, 2026, and surplus
  compensation offers in the €0.03–0.10/kWh range
- Ayuntamiento de Madrid fiscal ordinances — IBI bonus up to 50%, ICIO
  bonus up to 95%, including the provision for shared installations
- Ofgem price cap, 1 July – 30 September 2026 — 26.11 p/kWh electricity
  unit rate, used as the GB comparator
- Eurostat Housing 2025 and INE — 65% of the Spanish population in flats,
  73% in EU cities

**Geometry**
- OpenStreetMap via Overpass API — 100 building footprints, 450 × 450 m,
  Salamanca / Castellana, Madrid. © OpenStreetMap contributors, ODbL.

---

## 10. Repository layout

    config/madrid.yaml        every input, with source and confidence
    src/solar_geometry.py     sun position, from scratch
    src/irradiance.py         clear sky, day types, diffuse split, transposition
    src/shading.py            horizon profiles, beam masks, sky view factors
    src/roof_model.py         projection, OSM ingest, heights, panel packing
    src/panel.py              cell temperature, losses, degradation
    src/demand.py             Spanish load shapes and tariff periods
    src/economics.py          marginal capex, dispatch, RD 244/2019, returns
    run_madrid.py             the whole study in one command
    attribute_shading.py      what the shading model was actually worth
    build_web.py              assemble the self-contained page
    tests/test_sanity.py      theory, reference, consistency, regression
    tests/test_city.py        the index returns the same horizon as the scan
    tests/check_js_port.py    proves the map's sun matches the model's
    data/osm_raw.json         the Overpass extract, one block
    outputs/                  run logs and the JSON the maps read

    fetch_city.py             Overpass, tile by tile, cached and resumable
    run_city.py               the same model over central Madrid, in parallel
    analyse_city.py           block extrapolation vs city measurement
    build_city_web.py         assemble the city map
    src/spatial.py            the grid index that makes city scale possible
    web/index.html            the block map
    web/city/index.html       the city map

`web/city/index.html` is the single final page. It contains the write-up,
the city map, and a date/time-controlled 2D shadow projection. It needs its
`tiles/` directory beside it and must be served over http rather than opened
from `file://`, because the panel layer is fetched rather than inlined.

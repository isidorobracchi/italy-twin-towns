import * as maplibregl
    from "https://unpkg.com/maplibre-gl@^6.9.0/dist/maplibre-gl.mjs";


let municipalities = [];
let municipalityByLabel = new Map();

let map = null;
let markers = [];

let currentSource = null;
let currentMatch = null;


// --------------------------------------------------
// ELEMENTS
// --------------------------------------------------

const townInput =
    document.getElementById("town-input");

const townList =
    document.getElementById("town-list");

const findButton =
    document.getElementById("find-button");

const statusElement =
    document.getElementById("status");

const resultsElement =
    document.getElementById("results");

const modeSelect =
    document.getElementById("mode-select");

const excludeRegion =
    document.getElementById("exclude-region");

const minDistanceInput =
    document.getElementById("min-distance");

const peopleWeightInput =
    document.getElementById("people-weight");

const economyWeightInput =
    document.getElementById("economy-weight");

const territoryWeightInput =
    document.getElementById("territory-weight");

const peopleWeightValue =
    document.getElementById("people-weight-value");

const economyWeightValue =
    document.getElementById("economy-weight-value");

const territoryWeightValue =
    document.getElementById("territory-weight-value");

const resetWeights =
    document.getElementById("reset-weights");

const copyLinkButton =
    document.getElementById("copy-link");

const shareResultButton =
    document.getElementById("share-result");

const whatsappButton =
    document.getElementById("share-whatsapp");

const downloadStoryButton =
    document.getElementById("download-story");


// --------------------------------------------------
// FEATURES
// --------------------------------------------------

const PEOPLE_FEATURES = [
    "people_population",
    "people_density",
    "people_young",
    "people_elderly",
    "people_mean_age"
];


const ECONOMY_FEATURES = [
    "economy_income",
    "economy_taxpayers",
    "economy_high_income"
];


// --------------------------------------------------
// INIT
// --------------------------------------------------

async function init() {

    try {

        const response =
            await fetch(
                "data/municipalities.json"
            );

        if (!response.ok) {

            throw new Error(
                "Impossibile caricare i dati."
            );

        }

        municipalities =
            await response.json();

        buildSearchList();

        statusElement.textContent =
            `${municipalities.length.toLocaleString("it-IT")} comuni pronti al confronto`;

        restoreFromUrl();

    }

    catch (error) {

        console.error(error);

        statusElement.textContent =
            "Errore nel caricamento dei dati.";

    }

}


// --------------------------------------------------
// SEARCH
// --------------------------------------------------

function municipalityLabel(town) {

    return (
        `${town.name} — ${town.province_abbr} (${town.region})`
    );

}


function buildSearchList() {

    const fragment =
        document.createDocumentFragment();

    municipalities.forEach(town => {

        const label =
            municipalityLabel(town);

        municipalityByLabel.set(
            label.toLowerCase(),
            town
        );

        const option =
            document.createElement("option");

        option.value =
            label;

        fragment.appendChild(
            option
        );

    });

    townList.appendChild(
        fragment
    );

}


function resolveTown(input) {

    const value =
        input
            .trim()
            .toLowerCase();

    if (!value) {
        return null;
    }

    if (
        municipalityByLabel.has(value)
    ) {

        return municipalityByLabel.get(
            value
        );

    }

    const matches =
        municipalities.filter(
            town =>
                town.name
                    .toLowerCase()
                === value
        );

    if (
        matches.length === 1
    ) {

        return matches[0];

    }

    return null;

}


// --------------------------------------------------
// DISTANCES
// --------------------------------------------------

function rmsDistance(
    townA,
    townB,
    features
) {

    let total = 0;

    for (
        const feature
        of features
    ) {

        const difference =
            townA[feature]
            - townB[feature];

        total +=
            difference ** 2;

    }

    return Math.sqrt(
        total
        / features.length
    );

}


function territoryDistance(
    townA,
    townB
) {

    const area =
        Math.abs(
            townA.territory_area
            - townB.territory_area
        );

    const altitude =
        Math.abs(
            townA.territory_altitude
            - townB.territory_altitude
        );

    const coastal =
        townA.territory_coastal
        === townB.territory_coastal
            ? 0
            : 1;

    const degurba =
        Math.abs(
            townA.territory_degurba
            - townB.territory_degurba
        ) / 2;

    return (
        0.30 * area
        +
        0.30 * altitude
        +
        0.20 * coastal
        +
        0.20 * degurba
    );

}


function scaleDistance(
    townA,
    townB
) {

    return Math.abs(
        Math.log(
            townA.population_2026
            / townB.population_2026
        )
    );

}


function geographicDistance(
    townA,
    townB
) {

    const R = 6371;

    const lat1 =
        degreesToRadians(
            townA.latitude
        );

    const lat2 =
        degreesToRadians(
            townB.latitude
        );

    const dLat =
        degreesToRadians(
            townB.latitude
            - townA.latitude
        );

    const dLon =
        degreesToRadians(
            townB.longitude
            - townA.longitude
        );

    const a =
        Math.sin(
            dLat / 2
        ) ** 2
        +
        Math.cos(lat1)
        *
        Math.cos(lat2)
        *
        Math.sin(
            dLon / 2
        ) ** 2;

    return (
        R
        * 2
        * Math.atan2(
            Math.sqrt(a),
            Math.sqrt(1 - a)
        )
    );

}


function degreesToRadians(value) {

    return (
        value
        * Math.PI
        / 180
    );

}


function median(values) {

    const sorted =
        [...values]
            .sort(
                (a, b) =>
                    a - b
            );

    const middle =
        Math.floor(
            sorted.length / 2
        );

    if (
        sorted.length % 2
        === 0
    ) {

        return (
            sorted[middle - 1]
            +
            sorted[middle]
        ) / 2;

    }

    return sorted[middle];

}


// --------------------------------------------------
// WEIGHTS
// --------------------------------------------------

function getWeights() {

    let people =
        Number(
            peopleWeightInput.value
        );

    let economy =
        Number(
            economyWeightInput.value
        );

    let territory =
        Number(
            territoryWeightInput.value
        );

    if (
        people
        + economy
        + territory
        === 0
    ) {

        people = 1;
        economy = 1;
        territory = 1;

    }

    const total =
        people
        + economy
        + territory;

    return {

        people:
            people / total,

        economy:
            economy / total,

        territory:
            territory / total

    };

}


// --------------------------------------------------
// MODEL
// --------------------------------------------------

function calculateMatches(
    sourceTown
) {

    const raw =
        municipalities
            .filter(
                town =>
                    town.istat_code
                    !== sourceTown.istat_code
            )
            .map(
                town => ({

                    town,

                    people:
                        rmsDistance(
                            sourceTown,
                            town,
                            PEOPLE_FEATURES
                        ),

                    economy:
                        rmsDistance(
                            sourceTown,
                            town,
                            ECONOMY_FEATURES
                        ),

                    territory:
                        territoryDistance(
                            sourceTown,
                            town
                        ),

                    scale:
                        scaleDistance(
                            sourceTown,
                            town
                        ),

                    geographic:
                        geographicDistance(
                            sourceTown,
                            town
                        )

                })
            );


    const peopleMedian =
        median(
            raw
                .map(x => x.people)
                .filter(x => x > 0)
        );

    const economyMedian =
        median(
            raw
                .map(x => x.economy)
                .filter(x => x > 0)
        );

    const territoryMedian =
        median(
            raw
                .map(x => x.territory)
                .filter(x => x > 0)
        );

    const scaleMedian =
        median(
            raw
                .map(x => x.scale)
                .filter(x => x > 0)
        );


    const weights =
        getWeights();


    return raw.map(
        item => {

            const people =
                item.people
                / peopleMedian;

            const economy =
                item.economy
                / economyMedian;

            const territory =
                item.territory
                / territoryMedian;

            const scale =
                item.scale
                / scaleMedian;


            const categoryDistance =
                weights.people
                * people
                +
                weights.economy
                * economy
                +
                weights.territory
                * territory;


            const combined =
                0.85
                * categoryDistance
                +
                0.15
                * scale;


            return {

                ...item,

                peopleNormalized:
                    people,

                economyNormalized:
                    economy,

                territoryNormalized:
                    territory,

                scaleNormalized:
                    scale,

                combined

            };

        }
    );

}


// --------------------------------------------------
// FILTERING
// --------------------------------------------------

function filterMatches(
    matches,
    sourceTown
) {

    const minDistance =
        Number(
            minDistanceInput.value
        ) || 0;

    const mode =
        modeSelect.value;


    return matches.filter(
        match => {

            if (
                excludeRegion.checked
                &&
                match.town.region
                === sourceTown.region
            ) {

                return false;

            }


            let requiredDistance =
                minDistance;


            if (
                mode === "surprise"
            ) {

                requiredDistance =
                    Math.max(
                        requiredDistance,
                        400
                    );

            }


            return (
                match.geographic
                >= requiredDistance
            );

        }
    );

}


// --------------------------------------------------
// CHOOSE RESULT
// --------------------------------------------------

function chooseMatch(matches) {

    const mode =
        modeSelect.value;


    if (!matches.length) {
        return null;
    }


    if (
        mode === "opposite"
    ) {

        return [...matches]
            .sort(
                (a, b) =>
                    b.combined
                    - a.combined
            )[0];

    }


    if (
        mode === "surprise"
    ) {

        const candidates =
            [...matches]
                .sort(
                    (a, b) =>
                        a.combined
                        - b.combined
                )
                .slice(
                    0,
                    25
                );


        return candidates[
            Math.floor(
                Math.random()
                * candidates.length
            )
        ];

    }


    return [...matches]
        .sort(
            (a, b) =>
                a.combined
                - b.combined
        )[0];

}


// --------------------------------------------------
// SCORES
// --------------------------------------------------

function similarityScore(distance) {

    return Math.round(
        100
        * Math.exp(
            -distance
        )
    );

}


function differenceScore(distance) {

    return (
        100
        - similarityScore(
            distance
        )
    );

}


// --------------------------------------------------
// RUN
// --------------------------------------------------

function runMatch() {

    const sourceTown =
        resolveTown(
            townInput.value
        );


    if (!sourceTown) {

        statusElement.textContent =
            "Seleziona un comune dall'elenco.";

        return;

    }


    statusElement.textContent =
        "Sto confrontando 7.894 comuni...";


    const matches =
        calculateMatches(
            sourceTown
        );


    const filtered =
        filterMatches(
            matches,
            sourceTown
        );


    const match =
        chooseMatch(
            filtered
        );


    if (!match) {

        statusElement.textContent =
            "Nessun risultato con questi filtri.";

        return;

    }


    currentSource =
        sourceTown;

    currentMatch =
        match;


    renderResult(
        sourceTown,
        match
    );


    updateUrl(
        sourceTown
    );


    statusElement.textContent =
        `${filtered.length.toLocaleString("it-IT")} comuni confrontati`;

}


// --------------------------------------------------
// RESULT MODE
// --------------------------------------------------

function getModeContent() {

    const mode =
        modeSelect.value;


    if (
        mode === "opposite"
    ) {

        return {

            eyebrow:
                "IL TUO COMUNE OPPOSTO",

            card:
                "Comune opposto",

            breakdownEyebrow:
                "PERCHÉ SONO DIVERSI",

            breakdownTitle:
                "Profilo di differenza",

            scoreLabel:
                "diversità"

        };

    }


    if (
        mode === "surprise"
    ) {

        return {

            eyebrow:
                "UN GEMELLO INASPETTATO",

            card:
                "Comune sorpresa",

            breakdownEyebrow:
                "COSA LI AVVICINA",

            breakdownTitle:
                "Profilo del match",

            scoreLabel:
                "affinità"

        };

    }


    return {

        eyebrow:
            "IL TUO COMUNE GEMELLO",

        card:
            "Comune gemello",

        breakdownEyebrow:
            "PERCHÉ SI ASSOMIGLIANO",

        breakdownTitle:
            "Profilo di affinità",

        scoreLabel:
            "affinità"

    };

}


// --------------------------------------------------
// RENDER
// --------------------------------------------------

function renderResult(
    sourceTown,
    match
) {

    const twin =
        match.town;

    const mode =
        modeSelect.value;

    const copy =
        getModeContent();


    const score =
        mode === "opposite"
            ? differenceScore(
                match.combined
            )
            : similarityScore(
                match.combined
            );


    document.getElementById(
        "result-eyebrow"
    ).textContent =
        copy.eyebrow;


    document.getElementById(
        "twin-card-label"
    ).textContent =
        copy.card;


    document.getElementById(
        "breakdown-eyebrow"
    ).textContent =
        copy.breakdownEyebrow;


    document.getElementById(
        "breakdown-title"
    ).textContent =
        copy.breakdownTitle;


    document.getElementById(
        "score-label"
    ).textContent =
        copy.scoreLabel;


    document.getElementById(
        "match-title"
    ).textContent =
        `${sourceTown.name} ↔ ${twin.name}`;


    document.getElementById(
        "match-location"
    ).textContent =
        `${twin.province}, ${twin.region}`;


    document.getElementById(
        "match-score"
    ).textContent =
        `${score}%`;


    document.getElementById(
        "source-name"
    ).textContent =
        sourceTown.name;


    document.getElementById(
        "source-location"
    ).textContent =
        `${sourceTown.province}, ${sourceTown.region}`;


    document.getElementById(
        "source-stats"
    ).innerHTML =
        statsHtml(
            sourceTown
        );


    document.getElementById(
        "twin-name"
    ).textContent =
        twin.name;


    document.getElementById(
        "twin-location"
    ).textContent =
        `${twin.province}, ${twin.region}`;


    document.getElementById(
        "twin-stats"
    ).innerHTML =
        statsHtml(
            twin
        );


    renderCategoryBars(
        match
    );


    document.getElementById(
        "geo-distance"
    ).textContent =
        `${Math.round(
            match.geographic
        ).toLocaleString("it-IT")} km`;


    document.getElementById(
        "explanation"
    ).textContent =
        buildExplanation(
            sourceTown,
            twin,
            match
        );


    resultsElement.classList.remove(
        "hidden"
    );


    renderMap(
        sourceTown,
        twin
    );


    resultsElement.scrollIntoView(
        {
            behavior:
                "smooth",

            block:
                "start"
        }
    );

}


// --------------------------------------------------
// STATS
// --------------------------------------------------

function statsHtml(town) {

    return `

        <div class="stat">
            <span>Popolazione</span>
            <strong>
                ${Math.round(
                    town.population_2026
                ).toLocaleString("it-IT")}
            </strong>
        </div>

        <div class="stat">
            <span>Densità</span>
            <strong>
                ${Math.round(
                    town.density_km2
                ).toLocaleString("it-IT")}
                ab./km²
            </strong>
        </div>

        <div class="stat">
            <span>Reddito medio</span>
            <strong>
                €${Math.round(
                    town.average_income
                ).toLocaleString("it-IT")}
            </strong>
        </div>

        <div class="stat">
            <span>Età media</span>
            <strong>
                ${town.mean_age.toFixed(1)}
            </strong>
        </div>

        <div class="stat">
            <span>Superficie</span>
            <strong>
                ${town.area_km2.toFixed(1)}
                km²
            </strong>
        </div>

        <div class="stat">
            <span>Altitudine</span>
            <strong>
                ${Math.round(
                    town.altitude_m
                )} m
            </strong>
        </div>

    `;

}


// --------------------------------------------------
// BARS
// --------------------------------------------------

function renderCategoryBars(match) {

    const opposite =
        modeSelect.value
        === "opposite";


    const categories = [

        {
            label:
                "Persone",

            similarity:
                similarityScore(
                    match.peopleNormalized
                )
        },

        {
            label:
                "Economia",

            similarity:
                similarityScore(
                    match.economyNormalized
                )
        },

        {
            label:
                "Territorio",

            similarity:
                similarityScore(
                    match.territoryNormalized
                )
        }

    ];


    const container =
        document.getElementById(
            "category-bars"
        );


    container.innerHTML =
        categories
            .map(
                category => {

                    const score =
                        opposite
                            ? 100
                                - category.similarity
                            : category.similarity;

                    return `

                        <div class="category-row">

                            <div class="category-label">

                                <span>
                                    ${category.label}
                                </span>

                                <strong>
                                    ${score}%
                                </strong>

                            </div>

                            <div class="bar-track">

                                <div
                                    class="bar-fill ${opposite ? "difference" : ""}"
                                    style="
                                        width:
                                        ${score}%
                                    "
                                ></div>

                            </div>

                        </div>

                    `;

                }
            )
            .join("");

}


// --------------------------------------------------
// EXPLANATION
// --------------------------------------------------

function buildExplanation(
    source,
    twin,
    match
) {

    const opposite =
        modeSelect.value
        === "opposite";


    const categories = [

        {
            similarityName:
                "profilo demografico",

            differenceName:
                "struttura demografica",

            distance:
                match.peopleNormalized
        },

        {
            similarityName:
                "profilo economico",

            differenceName:
                "profilo economico",

            distance:
                match.economyNormalized
        },

        {
            similarityName:
                "caratteristiche territoriali",

            differenceName:
                "caratteristiche territoriali",

            distance:
                match.territoryNormalized
        }

    ];


    const populationDifference =
        Math.abs(
            twin.population_2026
            - source.population_2026
        );


    const incomeDifference =
        Math.abs(
            twin.average_income
            - source.average_income
        );


    if (opposite) {

        categories.sort(
            (a, b) =>
                b.distance
                - a.distance
        );

        const strongest =
            categories[0]
                .differenceName;


        return (
            `${source.name} e ${twin.name} emergono come due dei comuni più distanti nel modello, soprattutto per ${strongest}. `
            +
            `La popolazione differisce di ${Math.round(
                populationDifference
            ).toLocaleString("it-IT")} abitanti e il reddito medio dichiarato di circa €${Math.round(
                incomeDifference
            ).toLocaleString("it-IT")}. `
            +
            `Le percentuali sopra indicano il grado di differenza: più sono alte, più i due comuni sono lontani su quella dimensione.`
        );

    }


    categories.sort(
        (a, b) =>
            a.distance
            - b.distance
    );


    const strongest =
        categories[0]
            .similarityName;


    return (
        `${source.name} e ${twin.name} risultano particolarmente vicini per il loro ${strongest}. `
        +
        `La popolazione differisce di ${Math.round(
            populationDifference
        ).toLocaleString("it-IT")} abitanti e il reddito medio dichiarato di circa €${Math.round(
            incomeDifference
        ).toLocaleString("it-IT")}. `
        +
        `L'indice combina Persone, Economia e Territorio e tiene conto anche della differenza di scala demografica.`
    );

}


// --------------------------------------------------
// MAP
// --------------------------------------------------

function renderMap(
    source,
    twin
) {

    const sourceCoords = [
        source.longitude,
        source.latitude
    ];

    const twinCoords = [
        twin.longitude,
        twin.latitude
    ];


    if (!map) {

        map =
            new maplibregl.Map({

                container:
                    "map",

                style:
                    "https://tiles.openfreemap.org/styles/liberty",

                center:
                    sourceCoords,

                zoom:
                    5

            });


        map.addControl(
            new maplibregl.NavigationControl(),
            "top-right"
        );

    }


    markers.forEach(
        marker =>
            marker.remove()
    );

    markers = [];


    const sourceMarker =
        new maplibregl.Marker({
            color:
                "#153f35"
        })
            .setLngLat(
                sourceCoords
            )
            .setPopup(
                new maplibregl.Popup()
                    .setHTML(
                        `<strong>${source.name}</strong>`
                    )
            )
            .addTo(map);


    const twinMarker =
        new maplibregl.Marker({
            color:
                "#ff5f38"
        })
            .setLngLat(
                twinCoords
            )
            .setPopup(
                new maplibregl.Popup()
                    .setHTML(
                        `<strong>${twin.name}</strong>`
                    )
            )
            .addTo(map);


    markers.push(
        sourceMarker,
        twinMarker
    );


    const drawRoute = () => {

        const data = {

            type:
                "Feature",

            geometry: {

                type:
                    "LineString",

                coordinates: [
                    sourceCoords,
                    twinCoords
                ]

            }

        };


        if (
            map.getSource(
                "twin-route"
            )
        ) {

            map
                .getSource(
                    "twin-route"
                )
                .setData(
                    data
                );

        }

        else {

            map.addSource(
                "twin-route",
                {
                    type:
                        "geojson",

                    data
                }
            );


            map.addLayer({

                id:
                    "twin-route",

                type:
                    "line",

                source:
                    "twin-route",

                paint: {

                    "line-color":
                        "#ff5f38",

                    "line-width":
                        3,

                    "line-opacity":
                        0.8,

                    "line-dasharray":
                        [2, 2]

                }

            });

        }

    };


    if (
        map.isStyleLoaded()
    ) {

        drawRoute();

    }

    else {

        map.once(
            "load",
            drawRoute
        );

    }


    const bounds =
        new maplibregl.LngLatBounds();

    bounds.extend(
        sourceCoords
    );

    bounds.extend(
        twinCoords
    );


    map.fitBounds(
        bounds,
        {
            padding:
                90,

            maxZoom:
                8,

            duration:
                900
        }
    );


    setTimeout(
        () =>
            map.resize(),
        150
    );

}


// --------------------------------------------------
// STORY CARD
// --------------------------------------------------

async function createStoryBlob() {

    if (
        !currentSource
        ||
        !currentMatch
    ) {

        return null;

    }


    const canvas =
        document.createElement(
            "canvas"
        );


    canvas.width =
        1080;

    canvas.height =
        1920;


    const ctx =
        canvas.getContext(
            "2d"
        );


    const source =
        currentSource;

    const twin =
        currentMatch.town;

    const opposite =
        modeSelect.value
        === "opposite";


    const score =
        opposite
            ? differenceScore(
                currentMatch.combined
            )
            : similarityScore(
                currentMatch.combined
            );


    // Background

    ctx.fillStyle =
        "#f4f1e9";

    ctx.fillRect(
        0,
        0,
        1080,
        1920
    );


    // Decorative circles

    ctx.fillStyle =
        "#dbe8e1";

    ctx.beginPath();

    ctx.arc(
        950,
        130,
        300,
        0,
        Math.PI * 2
    );

    ctx.fill();


    ctx.fillStyle =
        "#ffe0d6";

    ctx.beginPath();

    ctx.arc(
        80,
        1770,
        350,
        0,
        Math.PI * 2
    );

    ctx.fill();


    // Brand

    ctx.fillStyle =
        "#153f35";

    ctx.font =
        "700 34px Arial";

    ctx.fillText(
        "COMUNE GEMELLO",
        80,
        110
    );


    // Main title

    ctx.fillStyle =
        "#171714";

    ctx.font =
        "700 72px Arial";

    ctx.fillText(
        opposite
            ? "Il mio comune opposto è"
            : "Il mio comune gemello è",
        80,
        300
    );


    ctx.fillStyle =
        "#ef5633";

    ctx.font =
        "700 118px Arial";


    wrapCanvasText(
        ctx,
        twin.name,
        80,
        440,
        920,
        125
    );


    // Source

    ctx.fillStyle =
        "#6d6b64";

    ctx.font =
        "36px Arial";

    ctx.fillText(
        `Partendo da ${source.name}`,
        80,
        720
    );


    // Score

    ctx.fillStyle =
        "#153f35";

    ctx.beginPath();

    ctx.arc(
        220,
        940,
        135,
        0,
        Math.PI * 2
    );

    ctx.fill();


    ctx.fillStyle =
        "#ffffff";

    ctx.textAlign =
        "center";

    ctx.font =
        "700 72px Arial";

    ctx.fillText(
        `${score}%`,
        220,
        935
    );


    ctx.font =
        "700 25px Arial";

    ctx.fillText(
        opposite
            ? "DIVERSITÀ"
            : "AFFINITÀ",
        220,
        985
    );


    ctx.textAlign =
        "left";


    // Stats

    ctx.fillStyle =
        "#171714";

    ctx.font =
        "700 38px Arial";

    ctx.fillText(
        `${Math.round(
            twin.population_2026
        ).toLocaleString("it-IT")} abitanti`,
        430,
        900
    );

    ctx.fillText(
        `€${Math.round(
            twin.average_income
        ).toLocaleString("it-IT")} reddito medio`,
        430,
        970
    );

    ctx.fillText(
        `${Math.round(
            currentMatch.geographic
        ).toLocaleString("it-IT")} km di distanza`,
        430,
        1040
    );


    // Categories

    const categories = [

        [
            "Persone",
            currentMatch.peopleNormalized
        ],

        [
            "Economia",
            currentMatch.economyNormalized
        ],

        [
            "Territorio",
            currentMatch.territoryNormalized
        ]

    ];


    let y =
        1220;


    categories.forEach(
        ([label, distance]) => {

            const similarity =
                similarityScore(
                    distance
                );

            const value =
                opposite
                    ? 100 - similarity
                    : similarity;


            ctx.fillStyle =
                "#171714";

            ctx.font =
                "700 32px Arial";

            ctx.fillText(
                label,
                80,
                y
            );


            ctx.fillText(
                `${value}%`,
                870,
                y
            );


            ctx.fillStyle =
                "#ddd8cd";

            roundRect(
                ctx,
                80,
                y + 30,
                880,
                24,
                12
            );

            ctx.fill();


            ctx.fillStyle =
                opposite
                    ? "#ef5633"
                    : "#153f35";


            roundRect(
                ctx,
                80,
                y + 30,
                880
                    * value
                    / 100,
                24,
                12
            );

            ctx.fill();


            y +=
                150;

        }
    );


    // Footer

    ctx.fillStyle =
        "#6d6b64";

    ctx.font =
        "29px Arial";

    ctx.fillText(
        "Scopri il tuo su Comune Gemello",
        80,
        1800
    );


    return new Promise(
        resolve => {

            canvas.toBlob(
                resolve,
                "image/png",
                1
            );

        }
    );

}


function wrapCanvasText(
    ctx,
    text,
    x,
    y,
    maxWidth,
    lineHeight
) {

    const words =
        text.split(" ");

    let line = "";
    let currentY = y;


    words.forEach(
        word => {

            const test =
                line
                + word
                + " ";

            if (
                ctx.measureText(
                    test
                ).width
                > maxWidth
                &&
                line
            ) {

                ctx.fillText(
                    line,
                    x,
                    currentY
                );

                line =
                    word + " ";

                currentY +=
                    lineHeight;

            }

            else {

                line =
                    test;

            }

        }
    );


    ctx.fillText(
        line,
        x,
        currentY
    );

}


function roundRect(
    ctx,
    x,
    y,
    width,
    height,
    radius
) {

    ctx.beginPath();

    ctx.roundRect(
        x,
        y,
        width,
        height,
        radius
    );

}


// --------------------------------------------------
// SHARE
// --------------------------------------------------

function shareText() {

    if (
        !currentSource
        ||
        !currentMatch
    ) {

        return "";

    }


    const target =
        currentMatch.town;


    if (
        modeSelect.value
        === "opposite"
    ) {

        return (
            `Ho scoperto il comune più diverso da ${currentSource.name}: ${target.name} 😅`
        );

    }


    return (
        `Ho scoperto il comune gemello di ${currentSource.name}: ${target.name} 👀`
    );

}


shareResultButton.addEventListener(
    "click",
    async () => {

        const blob =
            await createStoryBlob();


        const file =
            new File(
                [blob],
                "comune-gemello.png",
                {
                    type:
                        "image/png"
                }
            );


        const data = {

            title:
                "Comune Gemello",

            text:
                shareText(),

            url:
                window.location.href,

            files:
                [file]

        };


        try {

            if (
                navigator.share
                &&
                navigator.canShare
                &&
                navigator.canShare({
                    files:
                        [file]
                })
            ) {

                await navigator.share(
                    data
                );

                return;

            }


            if (
                navigator.share
            ) {

                await navigator.share({

                    title:
                        "Comune Gemello",

                    text:
                        shareText(),

                    url:
                        window.location.href

                });

                return;

            }


            await navigator.clipboard.writeText(
                window.location.href
            );

            alert(
                "Link copiato negli appunti."
            );

        }

        catch (error) {

            console.log(
                error
            );

        }

    }
);


whatsappButton.addEventListener(
    "click",
    () => {

        const message =
            `${shareText()}\n\n${window.location.href}`;


        const url =
            `https://wa.me/?text=${encodeURIComponent(
                message
            )}`;


        window.open(
            url,
            "_blank",
            "noopener,noreferrer"
        );

    }
);


downloadStoryButton.addEventListener(
    "click",
    async () => {

        const blob =
            await createStoryBlob();


        const url =
            URL.createObjectURL(
                blob
            );


        const anchor =
            document.createElement(
                "a"
            );


        anchor.href =
            url;

        anchor.download =
            "comune-gemello-story.png";


        anchor.click();


        URL.revokeObjectURL(
            url
        );

    }
);


copyLinkButton.addEventListener(
    "click",
    async () => {

        await navigator
            .clipboard
            .writeText(
                window.location.href
            );


        copyLinkButton.textContent =
            "✓ Copiato";


        setTimeout(
            () => {

                copyLinkButton.textContent =
                    "⧉ Link";

            },
            1300
        );

    }
);


// --------------------------------------------------
// URL
// --------------------------------------------------

function updateUrl(sourceTown) {

    const url =
        new URL(
            window.location.href
        );


    url.searchParams.set(
        "town",
        sourceTown.istat_code
    );

    url.searchParams.set(
        "mode",
        modeSelect.value
    );

    url.searchParams.set(
        "excludeRegion",
        excludeRegion.checked
            ? "1"
            : "0"
    );

    url.searchParams.set(
        "minDistance",
        minDistanceInput.value
    );

    url.searchParams.set(
        "people",
        peopleWeightInput.value
    );

    url.searchParams.set(
        "economy",
        economyWeightInput.value
    );

    url.searchParams.set(
        "territory",
        territoryWeightInput.value
    );


    window.history.replaceState(
        {},
        "",
        url
    );

}


function restoreFromUrl() {

    const params =
        new URLSearchParams(
            window.location.search
        );


    const code =
        params.get(
            "town"
        );


    if (!code) {
        return;
    }


    const town =
        municipalities.find(
            item =>
                item.istat_code
                === code
        );


    if (!town) {
        return;
    }


    townInput.value =
        municipalityLabel(
            town
        );


    if (
        params.has("mode")
    ) {

        modeSelect.value =
            params.get("mode");

    }


    excludeRegion.checked =
        params.get(
            "excludeRegion"
        ) === "1";


    if (
        params.has(
            "minDistance"
        )
    ) {

        minDistanceInput.value =
            params.get(
                "minDistance"
            );

    }


    if (
        params.has("people")
    ) {

        peopleWeightInput.value =
            params.get("people");

    }


    if (
        params.has("economy")
    ) {

        economyWeightInput.value =
            params.get("economy");

    }


    if (
        params.has("territory")
    ) {

        territoryWeightInput.value =
            params.get("territory");

    }


    updateWeightLabels();

    runMatch();

}


// --------------------------------------------------
// CONTROLS
// --------------------------------------------------

function updateWeightLabels() {

    peopleWeightValue.textContent =
        `${peopleWeightInput.value}%`;

    economyWeightValue.textContent =
        `${economyWeightInput.value}%`;

    territoryWeightValue.textContent =
        `${territoryWeightInput.value}%`;

}


[
    peopleWeightInput,
    economyWeightInput,
    territoryWeightInput
].forEach(
    input =>
        input.addEventListener(
            "input",
            updateWeightLabels
        )
);


resetWeights.addEventListener(
    "click",
    () => {

        peopleWeightInput.value =
            33;

        economyWeightInput.value =
            33;

        territoryWeightInput.value =
            34;

        updateWeightLabels();

    }
);


findButton.addEventListener(
    "click",
    runMatch
);


townInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key
            === "Enter"
        ) {

            runMatch();

        }

    }
);


// --------------------------------------------------
// METHODOLOGY
// --------------------------------------------------

const methodDialog =
    document.getElementById(
        "method-dialog"
    );


document.getElementById(
    "method-button"
).addEventListener(
    "click",
    () => {

        methodDialog.showModal();

    }
);


document.getElementById(
    "close-method"
).addEventListener(
    "click",
    () => {

        methodDialog.close();

    }
);


methodDialog.addEventListener(
    "click",
    event => {

        if (
            event.target
            === methodDialog
        ) {

            methodDialog.close();

        }

    }
);


// --------------------------------------------------
// START
// --------------------------------------------------

updateWeightLabels();

init();
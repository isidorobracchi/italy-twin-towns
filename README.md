# Comune Gemello

**Qual è il comune italiano più simile al tuo?**

Comune Gemello è un progetto data-driven che confronta i comuni italiani sulla base di caratteristiche demografiche, economiche e territoriali per individuare il comune con il profilo più affine — oppure quello più diverso.

🔗 **Live demo:** (https://isidorobracchi.github.io/italy-twin-towns/)

## Come funziona

Il progetto confronta 7.894 comuni italiani attraverso tre dimensioni principali:

### Persone

* popolazione residente
* densità abitativa
* quota di popolazione 0–14 anni
* quota di popolazione 65+
* età media

### Economia

* reddito medio dichiarato
* contribuenti ogni 100 residenti
* quota di contribuenti con reddito superiore a 55.000 €

### Territorio

* superficie comunale
* altitudine
* litoraneità
* grado di urbanizzazione (DEGURBA)

Le variabili quantitative vengono trasformate e normalizzate per renderle confrontabili. Le distanze vengono calcolate separatamente per ciascuna dimensione e successivamente combinate attraverso pesi configurabili dall'utente.

Il punteggio mostrato nell'app è un **indice di similarità**, non una probabilità.

## Modalità

**Il più simile** individua il comune con il profilo complessivamente più vicino.

**Il più diverso** cerca invece il comune più distante nello spazio delle caratteristiche considerate.

**Sorprendimi** seleziona uno dei migliori abbinamenti geograficamente lontani.

È inoltre possibile:

* escludere i comuni della stessa regione;
* impostare una distanza geografica minima;
* modificare il peso di Persone, Economia e Territorio;
* condividere il risultato tramite link o card social.

## Armonizzazione geografica

Le fonti utilizzate non condividono tutte lo stesso riferimento amministrativo.

La pipeline armonizza i dati alla geografia comunale italiana in vigore dopo le variazioni amministrative del 2026, per un totale di **7.894 comuni**.

In particolare sono state gestite esplicitamente:

* l'incorporazione di Lirio in Montalto Pavese;
* la fusione di Castegnero e Nanto nel nuovo comune di Castegnero Nanto.

Popolazione, struttura demografica, redditi, geometrie e indicatori derivati sono stati ricalcolati quando necessario invece di utilizzare semplici medie dei valori precedenti.

## Fonti

### ISTAT

* codici e unità amministrative territoriali;
* confini amministrativi comunali;
* popolazione residente per età e sesso;
* caratteristiche territoriali dei comuni;
* DEGURBA e classificazioni territoriali.

### Ministero dell'Economia e delle Finanze

* redditi e principali variabili IRPEF su base comunale, anno d'imposta 2024.

### Cartografia

La web app utilizza MapLibre GL JS e una basemap OpenFreeMap.

## Stack

### Data pipeline

* Python
* pandas
* GeoPandas
* NumPy
* scikit-learn

### Web

* HTML
* CSS
* JavaScript
* MapLibre GL JS

### Deployment

* GitHub Pages

## Struttura del progetto

```text
italy-twin-towns/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   └── pipeline e feature engineering
└── docs/
    ├── index.html
    ├── css/
    ├── js/
    └── data/
```

## Nota metodologica

Comune Gemello è un progetto esplorativo e divulgativo.

La similarità dipende dalle variabili selezionate, dalle trasformazioni applicate e dai pesi assegnati alle diverse dimensioni. Il risultato non deve quindi essere interpretato come una classificazione oggettiva o definitiva dei comuni italiani.

L'obiettivo del progetto è rendere esplorabili dati comunali complessi attraverso un'interfaccia semplice, giocosa e trasparente.

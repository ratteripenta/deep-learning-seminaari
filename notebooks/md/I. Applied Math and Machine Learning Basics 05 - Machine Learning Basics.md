
*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#5.-Machine-Learning-Basics" data-toc-modified-id="5.-Machine-Learning-Basics-1">5. Machine Learning Basics</a></span><ul class="toc-item"><li><span><a href="#5.1.-Learning-Algorithms" data-toc-modified-id="5.1.-Learning-Algorithms-1.1">5.1. Learning Algorithms</a></span><ul class="toc-item"><li><span><a href="#Tehtävä,-$T$" data-toc-modified-id="Tehtävä,-$T$-1.1.1">Tehtävä, $T$</a></span></li><li><span><a href="#Suorituskyky,-$P$" data-toc-modified-id="Suorituskyky,-$P$-1.1.2">Suorituskyky, $P$</a></span></li><li><span><a href="#Kokemus,-$E$" data-toc-modified-id="Kokemus,-$E$-1.1.3">Kokemus, $E$</a></span></li><li><span><a href="#5.1.1-Linear-Regression" data-toc-modified-id="5.1.1-Linear-Regression-1.1.4">5.1.1 Linear Regression</a></span></li></ul></li><li><span><a href="#5.2.-Capacity,-Underfitting-and-Overfitting" data-toc-modified-id="5.2.-Capacity,-Underfitting-and-Overfitting-1.2">5.2. Capacity, Underfitting and Overfitting</a></span></li><li><span><a href="#5.3.-Hyperaparameters-and-Validation-Sets" data-toc-modified-id="5.3.-Hyperaparameters-and-Validation-Sets-1.3">5.3. Hyperaparameters and Validation Sets</a></span></li><li><span><a href="#5.4.-Estimators,-Bias-and-Variance" data-toc-modified-id="5.4.-Estimators,-Bias-and-Variance-1.4">5.4. Estimators, Bias and Variance</a></span><ul class="toc-item"><li><span><a href="#Estimointi" data-toc-modified-id="Estimointi-1.4.1">Estimointi</a></span></li><li><span><a href="#Vääristymä" data-toc-modified-id="Vääristymä-1.4.2">Vääristymä</a></span></li><li><span><a href="#Varianssi" data-toc-modified-id="Varianssi-1.4.3">Varianssi</a></span></li><li><span><a href="#Bias-Variance-Tradeoff" data-toc-modified-id="Bias-Variance-Tradeoff-1.4.4">Bias-Variance Tradeoff</a></span></li></ul></li><li><span><a href="#5.5.-Maximum-Likelihood-Estimation" data-toc-modified-id="5.5.-Maximum-Likelihood-Estimation-1.5">5.5. Maximum Likelihood Estimation</a></span></li><li><span><a href="#5.6.-Bayesian-Statistics" data-toc-modified-id="5.6.-Bayesian-Statistics-1.6">5.6. Bayesian Statistics</a></span></li><li><span><a href="#5.7.-Supervised-Learning" data-toc-modified-id="5.7.-Supervised-Learning-1.7">5.7. Supervised Learning</a></span></li><li><span><a href="#5.8.-Unuspervised-Learning" data-toc-modified-id="5.8.-Unuspervised-Learning-1.8">5.8. Unuspervised Learning</a></span></li><li><span><a href="#5.9.-Stochastic-Gradient-Descent" data-toc-modified-id="5.9.-Stochastic-Gradient-Descent-1.9">5.9. Stochastic Gradient Descent</a></span></li><li><span><a href="#5.10.-Perinteiset-menetelmät-vs.-syväoppivat-menetelmät" data-toc-modified-id="5.10.-Perinteiset-menetelmät-vs.-syväoppivat-menetelmät-1.10">5.10. Perinteiset menetelmät vs. syväoppivat menetelmät</a></span></li></ul></li></ul></div>

## 5. Machine Learning Basics

Syväoppivat menetelmät rakentavat vahvasti perinteisempien koneoppimismenetelmien perustalle. Useimmilla malleilla on niihin spesifisti liittyviä parametreja, *hyperparametreja*, joilla voidaan säädellä mallin toimintaa (mm. oppimiskerroin). Koneoppimisessa sovelletaan tilastotiedettä, painopisteenään tietokoneiden käyttö monimutkaisten dataa kuvaavien funktioiden approksimoinnissa. Koneoppimismenetelmät voidaan pääpiirteittäin jakaa kahteen leiriin, ohjatusti oppiviin (*supervised learning*) ja ohjaamattomasti oppiviin (*unsupervised learning*). Useimmat menetelmät hyödyntävät optimointialgoritminaan stokasista jyrkimmän laskun menetelmää (*stochastic gradient descent*), jossa sana stokastinen viittaa dataa tuottavan prosessin satunnaisuuteen ja riippumattomuuteen sen todennäköisyysjakauman osalta (*i.i.d., independent and identically distributed*). Data määrittyy usein piirteiden (*feature*) perusteella, joita menetelmät oppivat painottamaan tai hyödyntämään kulloisenkin tehtävän mukaisesti. 

### 5.1. Learning Algorithms

Kirjassa koneoppimisalgoritmi määritellään seuraavasti:

> *"Koneoppimisalgoritmi on datasta oppimaan kykenevä algoritmi."*

Oppiminen määritellään seuraavasti:

> "*Algoritmin voidaan sanoa oppivan kokemuksesta $E$ (*experience*) tehtävän $T$ (*task*) mukaisesti suorituskyvyllä $P$ (*performance*), mikäli sen suorituskyky kasvaa kyseisessä tehtävässä kokemuksien myötä.*"

Koneoppimismallit pyrkivät parhaiten kuvailemaan niille syötetyt datasetit. Loogisesti tämä on mahdollista vain, mikäli mallilla on pääsy kaikkiin datasetin näytteisiin. Käytännnössä tämä on mahdotonta. Siksi koneoppimismallit pyrkivät ennemmin tarjoamaan datasetin kuvaamiseen liittyviä todennäköisyyksiä, jolloin jotkin opitut säännöt ovat todennäköisesti tosia suurimmalle osaa datasetin näytteistä. Koneoppimismallit ovat myös pääsääntöisesti käyttökelpoisia vain omissa kapeahkoissa käyttöalueissaan. Vaikka keskiarvolla mikään malli ei ole toistaan parempi, mikäli malleja tarkastellaan \*kaikkien mahdollisten\* todennäköisyysjakaumien ja tehtävien kohdalla, ei näin objektiivista tilannetta voida koskaan saavuttaa. Todellisuudessa eri ilmiöille voidaan löytää toisistaan eroavia todennäköisyysjakaumia, jolloin jotkut mallit voivat osoittautua toistaan paremmiksi.

#### Tehtävä, $T$

Koneoppimismenetelmiä voidaan käyttää ratkaisemaan erinäisiä tehtäviä hyvinkin kirjavasta joukosta. Näitä ovat ainakin:

 - **Luokittelu** (*classification*): Luokittelussa algoritmi pyrkii sijoittamaan sille annetun syötteen johonkin luokkaan. Näin vaikkapa hahmontunnistuksessa, jossa kuvista pyritään luokittelemaan niistä löytyvät esineet/asiat. Luokittelua voidaan tehdä myös puuttellisilla syötedatoilla, mutta tällöin yhden koko datasettiä kuvaavan funktion sijasta menetelmä pyrkii oppimaan joukon funktioita, jotka yhdessä kuvaavat datasettiä sen repaleisuudesta huolimatta.
 
 - **Ennustaminen** (*regression*): Tässä tehtävässä algoritmi pyrkii tuottamaan lukumuotoisen ennusteen perustuen sille annettuun syötedataan. Tehtävä on samankaltainen luokittelun kanssa, joskin opitun funktion arvojoukko on lukuja luokkien sijasta.
 
 - **Kuvailu** (*transcription*): Algoritmin tavoitteena on esimerkiksi tulkita kuva ja sen sisältö sanallisessa muodossa. Tätä voi olla esimerkiksi tekstin tunnistaminen kuvista, jossa koneelle luontaisesti lukukelvoton tietomuoto muutetaan koneluettavaksi - näin vaikkapa Google Street Viewin ja osoitenumeroiden tunnistuksen kohdalla.
 
 - **Konetulkkaus** (*machine translation*): Tässä tehtävässä algoritmille on opetettu joukko funktioita, jotka kykenevät tuottamaan tekstien käännöksiä kieleltä toiselle huomioiden kohde- ja lähdekielien kielioppisäännöt ja sanaston.
 
 - **Jäsentely** (*structured output*): Tehtävänä tämä on jo selkeästi laajempi. Esimerkiksi luonnollisen kielen prosessoinnissa (*natural language processing, NLP*) jäsentelyä voisi olla lauseiden jäsentäminen puurakenteeseen kieliopin perusteella. Toisaalta pikselitasolla tapahtuva kuvien luokittelu vaikkapa satelliittikuvista on myös jäsentämättömän tiedon jäsentämistä.
 
 - **Poikkeamien havaitseminen** (*anomaly detection*): Tässä algoritmin tehtävänä on oppia tavanomaiset ja sallitut tapahtumat ja samoin oppia erottelemaan näistä poikkeavat riittävällä varmuudella. Käytännössä tämä voi olla huijausten tunnistamista esimerkiksi luottoasioissa tai laitteiden toiminnassa.
 

#### Suorituskyky, $P$

Jotta algoritmi kykenee korjaamaan itseään, on sillä oltava käytettävissään koneoppimistehtävään sopiva suorituskykymittari. Luokittelun tapauksessa tämä mitta voi olla luokittelutarkkuus (*accuracy*) tai kääntäen virhetaso (*error rate*). Reaalilukujen tapauksissa on käytettävä niihin paremmin sopivia mittareita. Koska lähestulkoon kiinnostavinta koneoppimismalleissa on niiden yleistettävyys ennalta kohtaamattomiin syötteisiin (*generalization*), data jaetaan usein koulutussettiin (*training set*) ja testisettiin (*test set*). Sanomattakin lienee selvää, että menetelmien kirjon ohella myös suorituskykymittareita löytyy useita. Tärkeintä on määrittää tarkoin mitä tahdotaan mitata suhteutettuna valittuun tehtävään ja malliin.

#### Kokemus, $E$

Kokemuksella tarkoitetaan laajassa mielessä kaikkea sitä, mitä algoritmin annetaan kokea koulutuksen yhteydessä. Toisin sanoen kokemuksen käsite liittyy olennaisesti datasettiin, jolla ja jolle algoritmi koulutetaan. Datasetti ja oppimistehtävä yhdessä määrittävät vahvasti, mitä ja millä tavalla algoritmeille syötetään dataa. 

**Ohjaamattomassa oppimisessa** (*unsupervised learning*) algoritmille syötetään datasetti näyte kerrallaan. Algoritmin tehtäväksi jää sitten löytää yksittäisistä näytteistä koko datasettiä kuvaavia yleisiä piirteitä tai ominaisuuksia, jotka voidaan sitten mahdollisuuksien mukaan yleistää laajempaan, vielä havaitsemattomaan näytejoukkoon. Koska näytteistä puuttuu niin sanotut kohteet, ei algoritmia kouluteta ennustamaan tai luokittelemaan. Se voi kuitenkin ryhmitellä riittävän samankaltaisia näytteitä omiin joukkoihinsa(*clustering*) tai pyrkiä oppimaan datasetin taustalla vaikuttavat todennäköisyysfuntion.

**Ohjatussa oppimisessa** (*supervised learning*) algoritmille annetaan näytteiden lisäksi totuus kustakin näytteestä, eli luokka (*label*) tai tavoiteluku (*target*). Näin algoritmi opetetaan joko luokittelemaan tai ennustamaan pelkän dataan pohjautuvan ryhmittelyn sijasta; syötteeseen liitetty totuusarvo toimii opettajana. 

Täysin selkeää rajaa ei näiden kahden välille voida kuitenkaan vetää. Ohjaamattoman oppimisen ongelma voidaan pilkkoa yksittäisiksi ohjatun oppimisen ongelmiksi, kun taas ohjatun oppimisen ongelma voidaan ratkaista ensin oppimalla ongelmaan liittyvän datasetin todennäköisyysfunktio ohjaamattomasti ja sen jälkeen johtamalla siitä ohjatun oppimisen vaatimat kohdearvot. Muita rajatun datasetin oppimiskategorioita ovat mm. osittain ohjattu oppiminen (*semi-supervised learning*), jossa kohdearvot on tiedossa vain osalla näytteistä. Moni-instanssisessa oppimisessa (*multi-instance learning*) kokonaisia joukkoja on kerralla luokiteltu, joskaa yksittäisillä näytteillä tätä tietoa ei ole talletettuna.

Datasetti voi olla rajatun lisäksi myös jatkuvasti muuttuva. Tällöin esimerkiksi ollaan **vahvistuoppimisen** (*reinforcement learning*) alueella. Tällöin algoritmit ovat vuorovaikutuksessa ympäristönsä kanssa saaden sieltä palautetta korjaten itseään palautteen perusteella ja toimien sen jälkeen parhaimman päivitetyn tietonsa perusteella. Toisin sanoen kokemukset ja oppiminen ovat takaisinkytketyt toisiinsa.

Datasetit on tavallista kuvata matriisimuodossa, jossa sarakkeet vastaavat datasetin piirteitä ja rivit yksittäisiä näytteitä eli havaintoja. Tällöin datasetti voidaan esittää esimerkiksi matriisina $X \in \mathbb{R}^{155 \times 4}$, jolloin datasetissä on neljä piirrettä ja 155 havaintoa.

#### 5.1.1 Linear Regression

Konkretisoiva esimerkki koneoppimisesta on lineaarinen regressio (*linear regression*), jollaa voidaan ratkaista regression eli ennustamisen ongelmia; algoritmimme $T$ on regressio. Yksinkertaisimmillaan tavoitteena on rakentaa järjestelmä, joka saadessaan syötteenä vektorin $x \in \mathbb{R}^n$ antaa tulokseksi skalaariarvon $y \in \mathbb{R}$. Kosksa kyseessä on lineaarinen regressio, menetelmän tulokset ovat syötteiden lineaarikombinaatioita. 

Mallin tuottama ennuste $\hat{y}$ määritellään seuraavasti:

$$ \hat{y} = w^Tx,$$

jossa $w \in \mathbb{R}^n$ on parametri- tai painovektori. 

Tällä painovektorilla painotetaan jokainen vektorin $x$ alkio. Siksi se on tämän yksinkertaisen algoritmin koulutuksen kohteena. Jotta mallia voidaan kouluttaa tarvitaan vielä ainakin mittari mallin suorituskyvyn arvioimiseksi. Mittariksi kelpaa virheiden neliöiden keskiarvo (*mean squared error, MSE*); $P$ on virheiden neliöiden keskiarvo. Datasetti jaetaan kahtia koulutus- ja testisetteihin. Siinä tulee myös olemaan tiedot kohdearvoista; $E$ on ohjatun oppimisen alueella.

Kun määriteltyä suorituskykymittarifunktiota $P$ minimoidaan päivittämällä mallin painovektoria $w$ iteratiivisesti, päästään iteraatio kerrallaan kohti testisettiin nähden mahdollisimman tarkkoja ennusteita.

### 5.2. Capacity, Underfitting and Overfitting

Koneoppimisen ydinhaasteena on jo aiemminkin mainittu yleistettävien (*generalizing*) mallien tuottaminen. Kuten jo aiemminkin esitelty, tätä tavoitellaan jakamalla datasetit koulutus- ja testisetteihin. Mallia ei ikinä kouluteta testisetillä, mutta sen suorituskykyä mitataan perinteisesti vain sillä. Tilastollisen oppimisteoriaan (*statistical learning theory*) pohjaten lähtökohtana on, että sekä koulutus- että testisettien näytteet ovat pohjimmiltaan saman datantuottoprosessin (*data-generating process*) tuotoksia ja täyttävät rippumattomuuden ja yhtenevän taustajakauman (*i.i.d*) oletukset. Tällöin näiden kahden näytejoukon taustalla oleva datatuottojakauma on yhtenevä. 

Lähtökohtaisesti siis satunnaisesti valitun koulutussetin näytteen virheen ei pitäisi poiketa testisetin näytteen virheestä keskimäärin. Tällöin koneoppimisalgoritmien tavoitteet saadaan yksinkertaistettua seuraavasti:

 1. Pienennä koulutusvirhettä.
 2. Kavenna koulutus- ja testivirheiden eroa.
 

Mikäli algoritmi ei kykene pienentämään koulutusvirhettä riittävästi, se alisovittuu (*underfit*) suhteessa koulutussettiin. Jos taas algoritmi oppii koulutussetin liian hyvin, eli koulutus- ja testivirheiden välillä on merkittävä ero, algoritmi ylisovittuu (*overfit*) suhteessa koulutussettiin. Malliin liittyen tätä tasapainoa voidaan säädellä vaikuttamalla mallin kapasiteettiin (*capacity*), siihen, kuinka laajan kirjon erilaisia funktioita malli saa hyödyntää oppiessaan. Tätä kutsutaan tarkemmin edustavaksi kapasiteetiksi (*representational capacity*). 

Lineaarisen regression ongelmassa kapasiteetin lisääminen tarkoittaisi esimerkiksi polynomien sallimista lineaarikombinaatioiden lisäksi, jolloin kullekin näytteen alkiolle etsittäisin alkion kertoimen lisäksi kerroin sen neliölle. Esimerkkikaavana se näyttäisi tältä:

$$ \hat{y} = b + w_1x + w_2x^2 .$$

Mikäli mallin kapasiteetti on liian suuri, se ylisovittuu helposti. Liian pienellä kapasiteetilla se taas alisovittuu. Tämä on vielä visualisoitu alla yksinkertaisella koodilla, jossa satunnaisesti generoituun dataan sovitetaan eriasteisia suoria. Oikeinpuolimmaisin (`deg=7`) selkeästi ylisovittuu kohinaan, kun taas vasemmanpuoleisin (`deg=1`) ei löydä datasta mitään varsinaisesti yleistä kulmakerrointa kummempaa.


```python
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = 18, 4

data_x = np.arange(1, 11)
data_y = np.random.rand(10) * 5
x = np.arange(1, 101) / 10

for i in range(4):

    deg = (2 * i + 1)
    w = np.polyfit(data_x, data_y, deg=deg)
    p = np.poly1d(w)
    pred_y = p(x)

    plt.subplot(1, 4, i + 1)
    plt.title("deg={}".format(deg))
    plt.scatter(data_x, data_y)
    plt.plot(x, pred_y,)
    plt.xlim(0, 10)
    plt.ylim(0, 5)

plt.show()
```


![png](I.%20Applied%20Math%20and%20Machine%20Learning%20Basics%2005%20-%20Machine%20Learning%20Basics_files/I.%20Applied%20Math%20and%20Machine%20Learning%20Basics%2005%20-%20Machine%20Learning%20Basics_29_0.png)


Edustavan kapasiteetin suhteen optimaalisen ratkaisun löytäminen on kuitenkin useimmiten työlästä. Muuttamalla mallin parametreja ei löydetä välttämättä parasta mallia, vaan vain malli, joka pienentää koulutuksen aikaista virhettä merkittävästi. Kun huomioidaan mallin optimoinnin haasteet, voidaan alkaa puhua käytännöllisestä kapasiteetista (*effective capacity*), joka on pienempi tai enintään yhtä suuri edustavan kapasiteetin kanssa. 

Käytännössä tällä erottelulla tahdotaan ilmaista lausetta

> *"Monimutkaisempi ei tarkoita aina parempaa"*

tai toisin kääntäen

> *"Yksinkertaisempi on parempi"* (Occamin partaveitsi)

Yksiselitteisestä tämä ei kuitenkaan ole. Vaikka tilastollisen oppimisteorian puolella on kehitetty keinoja optimaalisen mallin kapasiteetin määrittämiseen esimerkiksi luokittelun ongelmissa, syväoppivat menetelmät eivät näistä keinoista juuri hyödy. Mallin ja sen parametrien sijasta käytännöllisen kapasiteetin pullonkaulaksi muodostuu optimointialgoritmit, joiden teoreettisen ymmärryksen taso on vielä riittämättömällä tasolla optimikapasiteetin määrittelyyn. Toisin sanoen syväoppivien menetelmien kohdalla on oltava riittävästi kompleksisuutta, jotta koulutuksen aikainen testivirhe saadaan riittävän pieneksi.

Eräs mallin sisäisen rakenteen huomiotta jättävä keino kapasiteettiin vaikuttamiseksi on regularisoinnin (*regularizer*) käyttöönotto. Käytännössä regularisoinnilla tarkoitetaan mallin parametrien ajan mittaan tapahtuvaa heikentymistä (*decay*). Esimerkiksi edellisen esimerkin 7-asteinen malli pysyy kautta linjan samalla, mutta sen painokertoimiin vaikutetaan mahdollisesti jopa vaimentamalla suurin osa niistä jopa kokonaan regularisointikertoimella $\lambda$. Mikäli $\lambda$ on suuri, vain merkittävin kerroin jää vaikuttavaksi. Matala regularisointikerroin taas sallii suurempaa monimutkaisuutta. Itse malli pysyy kuitenkin koko ajan samana. Yleisemmin regularisointi määritellään kaikiksi niiksi toimiksi, joilla vaikutetaan testivirheeseen koulutusvirheen jääden koskemattomaksi.

### 5.3. Hyperaparameters and Validation Sets

Hyperparametrit (*hyperparameters*) mainittiin jo tämän luvun alussa. Niillä tarkoitetaan mallin toimintaan vaikuttavia parametreja, joita säätämällä mallia voidaan saada paremmin sovitettua dataan. Nämä parametrit ovat niitä arvoja, joihin tukeutuen mallit oppivat. Niitä ei optimoida koulutuksen aikana mitenkään, vaan ne vaikuttavat optimointiin. Tosin, kuten aiemmassa polynomisen regression esimerkissä, näitä parametreja on mahdollista optimoida etsimällä arvot, jotka tuottavat pienimmän virheen mallin. Mikäli esimerkiksi koneoppimismalli sais itse määrittää oman kapasiteettinsa, se päätyisi aina suurimpaan mahdolliseen kapasiteettin koulutusdatan osalta.

Jotta näin ei kuitenkaan pääse tapahtumaan, on data hajautettava koulutus- ja testidatan lisäksi vielä erilliseen validointisettiin. Koska koulutuksessa käytetään testisettiä testivirheen laskentaan ja täten mallin sisäisten parametrien päivitykseen, on datasetistä erotettava vielä erillinen oma osansa, jolla lopullinen yleistysvirhe (*generalization error*) mitataan. Tämä mittaus kertoo kulloinkin käytetyillä hyperparametreilla alustetun mallin suorituskyvystä. Käytännössä kyseessä on edelleen testivirhe, mutta tätä virhettä ei käytetä mallin kouluttamiseen mitenkään. Siksi on myös tärkeää, että validointisetin näytteitä ei ole käytetty mallin kouluttamisessa, ts. koulutuksen aikaisen testivirheen tuottamisessa. 

Käytännössä tätä varten kehitetty menetelmä tunnetaan nimellä ristiinvalidointi (*cross-validation, CV*). Ristiinvalidoinnissa datasetti $D^{m \times n}$ jaetaan heti alussa koulutus- ja testisetteihin. Tämän jälkeen malli koulutetaan *vain* koulutusdatalla, joka ositellaan sisäisesti koulutus- ja testisetteihin. 

Koulutussetti $T$ ja validointisetti $V$ voidaan muodostaa esimerkiksi siten, että

$$ T = 0.8n_D $$
$$ V = 0.2n_D, $$

jolloin koulutussetissä $T$ on 80% datasetin riveistä ja validointisetissä $V$ 20%. 

Koulutusta varten koulutussetti jaotellaan edelleen, esimerkiksi kymmeneen osaan. Tämän jälkeen koulutus suoritetaan siten, että jokainen osa toimii kerran testisettinä. Alla on sama esitetty taulukkona, jossa yksi rivi vastaa yhtä koulutuskertaa ja *t* tarkoittaa, että kyseinen koulutussetin osa toimii koulutuksen testisettinä.

| $t_1$ | $t_2$ | $t_3$ | $t_4$ | $t_5$ | $t_6$ | $t_7$ | $t_8$ | $t_9$ | $t_{10}$ |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
|   t   |       |       |       |       |       |       |       |       |      |   
|       |   t   |       |       |       |       |       |       |       |      |    
|       |       |   t   |       |       |       |       |       |       |      |   
|       |       |       |   t   |       |       |       |       |       |      |   
|       |       |       |       |   t   |       |       |       |       |      |   
|       |       |       |       |       |   t   |       |       |       |      |   
|       |       |       |       |       |       |   t   |       |       |      |   
|       |       |       |       |       |       |       |   t   |       |      |   
|       |       |       |       |       |       |       |       |   t   |      |   
|       |       |       |       |       |       |       |       |       |   t  |

Koulutuksen testivirhe lasketaan tällöin näiden koulutuskertojen keskiarvona.

### 5.4. Estimators, Bias and Variance

> *Malli $\ne$ estimaattori*

#### Estimointi

Estimaattorit liittyvät kiinteästi koneoppimiseen. Estimaattoreilla tarkoitetaan sellaisia funktioita, joilla pyritään löytämään parhaimmat malliin ja dataan yhdessä sopivat parametrit (esim. painot). Piste-estimaattoreilla (*point-estimator*) pyritään löytämään yksittäinen paras parametrien kokoonpano, joilla malli tuottaa kelvollisen tuloksen. Joissain tapauksissa parametrien sijasta estimoinnin kohteena voi olla kokonainen ennusteita tuottava funktio. Tällöinkin kyseessä on piste-estimoitiin palautettava ongelma, jossa piste on yksinkertaisesti yksittäinen funktio mahdollisten funktioiden joukossa tai avaruudessa.

Yksinkertaistaen voitaisiin todeta, että estimaattorilla tarkoitetaan kohdefunktiota, jolla saadaan tietoa mallin suorituskyvystä. 

Esimerkki estimaattorista on esimerkiksi $k$-kertainen ristiinvalidointialgoritmi (*$k$-fold cross-validation*), jota voidaan käyttää koneoppimismallin testivirhettä normaalia pienemmän datasetin kanssa. Kun algoritmiin syötetään samalle datasetille koulutetut koneoppimismallit, se antaa kummallekin keskivirheen ja keskihajonnan, joidenka perusteella voidaan päätellä kahdesta mallista parempi.

#### Vääristymä

Estimaattorin sanotaan olevan vääristynyt (*biased*), mikäli sen odotusarvo poikkeaa datasetin odotusarvosta; estimaattorin ja datasetin jakaumien ero näkyy vääristymänä (*bias*). Mikäli estimaattorin ja datasetin odotusarvot ovat samat, mallia voidaan nimittää vääristymättömäksi (*unbiased*). Vaikkakin vääristymättömät estimaattorit vaikuttavatkin lähtökohtaisesti toivotuilta, vääristyneillä estimaattoreilla on joitakin yllättäviä toivottuja ominaisuuksia, minkä vuoksi niitä käytetään useammin. Näistä lisää myöhemmin.

#### Varianssi

Vääristymän lisäksi malleille voidaan laskea myös varianssi (*variance*). Käytännössä tämä tarkoittaa ihan samaa asiaa, kuin tilastotieteellinen varianssi, eli kuinka suuri on hajonta mallin keskiarvon ympärillä. Kuten tilastotieteessä, käytetyin hajonnan suure on keskihajonta. Datasettiä kasvattamalla on normaalisti mahdollisuus pienentää mallin varianssa *eli* lisätä sen tarkkuutta.

#### Bias-Variance Tradeoff

Näitä kahta eri suunnista mallin virhettä määrittelevää suuretta, vääristymää ja varianssia, voidaan esittää tikkataulun avulla. Kun mallin vääristymä ja varianssi ovat mahdollisimman pieniä, tikat osuvat tiiviisti taulun keskialueelle. Korkean varianssin malli osuu keskimäärin keskelle, mutta tikat hajoavat laajemmalle ja pisteet jäävät näin pienemmäksi. Korkean vääristymän ja matalan varianssin malli saa tikat tiiviisti samalle alueelle, mutta tämä alue jää kauas taulun keskipisteestä. Korkean vääristymän ja varianssin malli osuu keskimäärin kunnolla ohi ja tikat menevät minne sattuu. 

Koneoppimismallien kanssa ollaan huomattu, että kasvattamalla mallin kapasiteettia vääristymä pienenee ja varianssi kasvaa. Tätä kutsutaan vääristymän ja varianssin vaihtokaupaksi (*bias-variance tradeoff*). 

Usein koneoppimisen yhteydessä käytetty estimaattori tunnetaan nimellä keskimääräinen neliövirhe (*mean squared error, MSE*). Se lasketaan keskiarvona vääristymän neliön ja varianssin summasta. 

### 5.5. Maximum Likelihood Estimation

Vääristymän ja varianssin tarkastelu jokaisen dataan sovitetun mallin kohdalla on verraten työlästä. Kuten ristiinvalidointiesimerkistä käy ilmi, se on tehtävä vertaillen kofiguraatioita toinen toiseensa kerta toisensa jälkeen. Suoraviivaisempaan optimaalisten parametrien selvittämiseen sopii paremmin suurimman todennäköisyyden estimointi (*maximum likelihood estimation, MLE*). 

Estimaattorille syötetään datasetin lisäksi joukko keskenään vertailtavia funktioita, joista kukin on opetettu datasetille $x$ ja omaa parametrit $\theta$. Näistä funktioista sitten valitaan paras. Parhaus määritellään sillä, kuinka samankaltainen mallin tuottama jakauma on datasetin jakaumaan nähden. Ero lasketaan hyödyntämällä aiemmin esiteltyä KL-divergenssiä, jossa vertailtavat todennäköisyysjakaumat ovat datasetin sisäinen jakauma ja mallin tuottama jakauma.

Useammin on laskennallisesti järkevää kuitenkin käyttää logaritmia, jolloin liian pieniin lukuihin helposti kaatuvat kertolaskut saadaan muunnettua summauksiksi. Tästä juontaakin mm. koneoppimiskirjastoista tuttu kohdefunktion nimi *log-likelihood*.

MLE on yleistetävissä ehdolliseksi lausekkeeksi, jossa etsitään suurimman ehdollisen logaritmisen todennäköisyyden (*conditional log-likelihood*) tuottavat parametrit:

$$ \theta_{ML} = \mathop{\arg\max}\limits_{\theta} P(Y \mid X;\theta). $$

MLE on ensisijaisesti piste-estimaattori. Merkinnällä $\mathop{\arg\max}\limits_{\theta}$ tarkoitetaan syötettä, joka maksimoi parametrien $\theta$ arvot.

### 5.6. Bayesian Statistics

Tähän asti käsitellyt parametrien estimointimenetelmät ovat keskittyneet frekvenssisiin tilastollisiin menetelmiin (*frequentist statistics*), joissa pyritään määrittämään yksi paras parametrien arvojoukko $\theta$. Bayesin tilastollisissa menetelmissä (*Bayesian statistics*) otetaan huomioon kaikki mahdolliset $\theta$ parametrien joukot. Frekvenssisisssä menetelmissä lähtökohtana on, että $\theta_{true}$ on määritelty mutta tuntematon ja estimaatti $\hat{\theta}$ pohjaa datasettiin ja on täten käsiteltävissä satunnaismuuttujana. Bayesin menetelmissä todennäköisyyksiä käytetään tiedon varmuuden tasojen kuvaamiseen. Datasetti havaitaan sellaisenaan eikä sitä käsitellä täten satunnaisena. Todellinen $\theta_{true}$ käsitellään määrittelemättömänä ja esitetään satunnaismuuttujana.

Ennen etenemistä valitaan riittävän laveasti kattava alustava todennäköisyysjakauma (*prior probability distribution*) esimerkiksi määrittämällä arvoalueen rajat tai olettamalla arvojen olevan tasaisesti jakautuneita. Tämän jälkeen alustavaa jakaumaa testataan dataa vasten. Prosessin myötä todennäköisimmät $\theta$:n arvot nousevat esiin muita arvoja todennäköisempinä. Menetelmä on mallinnuksen kannalta robusti, mutta laskennallisesti vaativa.

Tämä parametrien estimointimenetelmä eroaa MLE:stä kahdella tapaa:

 1. Piste-estimaation sijaan *$\theta$ estimoidaan koko sen mahdollisen jakauman avulla*.
 2. *Alustava todennäköisyysjakauma vaikuttaa lopullisiin parametreihin*, jolloin jakaumasta nousevat parametrien arvot ovat jonkseenkin alussa valitun alustavan jakauman tapaan tasaisesti jakautuneita.
 

Bayesin menetelmästä on mahdollista saada ulos myös selkeämpi piste-estimaatti. Tätä menetelmää kutsutaan *maximum a posteriori (MAP)* piste-estimaatiksi. Tällöin valitaan suurimman todennäköisyyden tuottava parametrijoukko $\theta$ ja jätetään jakauma muilta osin huomiotta.

### 5.7. Supervised Learning

Ohjatussa oppimisessa on kyse malleista, jotka oppivat kuvaamaan (*map*) syötteen $x$ ulostuloksi $\hat{y}$ oikein arvojen $y$ avulla. Todennäköisyyksiin pohjaavassa ohjatussa oppimisessa (*probabilistic..*) tähdätään ehdollisen todennäköisyyden $p(y \mid x)$ tuottamiseen, mikä on saavutettavissa esimerkiksi MLE:llä. Ohjatun oppimisen menetelmiä voidaan käyttää sekä regressioon että luokitteluun liittyvissä tehtävissä. Tärkeintä on, että syötedatoille löytyy lähtökohtaisesti aina ns. oikea tieto, eli joko luokka, arvo tai arvojoukko, joka kyseisellä syötteellä on havaittu.

Eräs tehokas esitelty malli on tukivektorikone (*support vector machine*), joka todennäköisyyksien sijaan tuottaa binäärisiä luokittelutietoja. Luokittelu perustuu datapisteet läpäisevän suoran käyttämiseen pisteiden luokittelussa. Muita perinteisiä esiteltyjä ohjattua oppimista hyödyntäviä koneoppimismalleja ovat mm. päätöspuut (*decision trees*) ja lähimpien naapurien menetelmä (*$k$-nearest neighbours*).

### 5.8. Unuspervised Learning

Vaikka tarkkaan ottaen ohjaamattoman ja ohjatun oppimisen välinen rajaviiva on häilyvä, eroavat oppimistavat toisistaan kohdetietojen osalta. Ohjaamattomassa oppimisessa syötteille ei ole oikeita vastauksia, vaan kyseistä oppimisstrategiaa hyödyntäviä menetelmiä käytetään datasta itsestään löytyvien asioiden tutkintaan. Perinteinen ohjaamattoman oppimisen ongelma sellaisen datasettia kuvaavan mallin löytäminen, joka säilyttäisi mahdollisimman paljon alkuperäisestä informaatiosta mutta olisi kuitenkin alkuperäistä datasettiä yksinkertaisempi. Yksinkertaisemmmalla tarkoitetaan joko vähempipiirteistä (*lower-dimensional*), harvaa (*sparse*) tai riippumatonta (*independent*) datasetin kuvausta mallin avulla.

Aiemmin esitelty PCA-algoritmi on hyvä esimerkki piirteiden määrää karsivasta ohjaamattoman oppimisen menetelmästä. Toinen tällä alueella paljon käytetty menetelmä on *$k$-means clustering*, jossa datasetti jaetaan $k$ joukkoon eli klusteriin. Menetelmä on iteratiivinen ja se pyrkii löytämään sellaiset datan sisäiset joukot, jotka ovat stabiileja. Joukot ovat stabiileja, kun klustereiden pisteiden keskiarvot eivät muutu enää niin paljoa, että klusterien koot muuttuisivat datapisteiden suhteen. Klusteroinnissa on kuitenkin haasteena, että eri klusterointimenetelmät tai jopa -kerrat voivat klusteroida dataa eri piirteiden perusteella.

### 5.9. Stochastic Gradient Descent

Syväoppivat menetelmät ovat pääsääntöisesti kaikki koulutettavissa stokastisella eli satunnaisuuteen pohjaavalla jyrkimmän laskun menetelmällä (*stochastic gradient descent, SGD*). Se on laajennos aiemmin esitellystä *gradient descentistä*. SGD:n pohjana toimivalla perinteisellä menetelmällä laskettuna datasetin kasvaessa kasvaa myös laskennan tarve lineaarisesti, mikä muodostuu rajoittavaksi tekijäksi. SGD lähestyy tätä ongelmaa datasetistä muodostettujen pikkuerien (*minibatch*) avulla. Mallin parametreja päivitetään vain erän tultua kokonaan käsitellyksi. Kun eräkoot ovat yhdestä muutamaan sataan näytteeseen, onnistuu suurenkin datasetin sovittaminen malliin laskennallisesti kevyemmin.

SGD muodostaa eräkohtaiset gradienttien keskiarvot. Näin toimiessa ei välttämättä päästä aina edes lokaaliin minimiin, mutta kohdefunktion riittävän pieni vähentyminen onnistuu näin käyttkelpoisen lyhyessä ajassa. Siitä syystä syväoppivat menetelmät hyödyntävät tätä optimointistrategiaa. Syväoppivien mallien lisäksi muutkin mallit hyödyntävät SGD:tä, kuten suuret lineaariset mallit.

### 5.10. Perinteiset menetelmät vs. syväoppivat menetelmät

Vaikka koneoppimisalgoritmin kokoaminen on verrattain helppoa sen koostuessa itsenäisistä osista kuten datasetti, malli, optimointimenetelmä ja kohdefunktio, niin perinteisillä menetelmillä on perustavanlaatuisia ongelmia tekoälyyn liittyvien tehtävien toteuttamisessa. Näitä ovat mm. puheen tai hahmontunnistus. Näihin ongelmiin syväoppivat menetelmät ovat tarjonneet käytännössä todennettuja toimivia ja suorituskykyisiä ratkaisuja, minkä vuoksi niiden käyttöönotto on viime vuosina räjähdysmäisesti kasvanut.

Eräs merkittävä koneoppimisen haaste on piirteiden kirous (*curse of dimensionality*). Kun datasetin piirreavaruus laajenee, kasvavat eri piirteiden mahdolliset konfiguraatiot eksponentiaalisesti. Matalapiirteisen datasetin kuvaaminen on vielä helppoa, kuten on yksi-, kaksi- tai kolmipiirteisen datasetin kohdalla. Kun datasetin piirremäärä kasvaa, alkaa datasetti olla piirrekohtaisesti kuitenkin sekä vaikeampaa kuvata että etenkin *harvempaa*. Tällöin datasetin piirreavaruudessa on lähtökohtaisesti enemmän aukkoja kuin dataa, jolloin dataa kuvaavan mallin kouluttaminen osoittautuu hankalaksi.

Jotta koneoppimismenetelmät toimisivat hyvin, niiden kohdalla on tehtävä oppimista ohjaavia ennakkopäätöksiä ("minkälaista funktiota opetellaan?", "miten optimoidaan?", jne.). Nämä päätökset määrittävät mm. sen, kuinka tasainen opittavan funktion on oltava (ks. Bayesian statistics). Näin toimittaessa tekoälyn alueella olevat ongelmat jäävät kuitenkin vaille ratkaisua. Ratkaisemattomuus ei kuitenkaan ole kiinni niinkään väärästä ennakkopäätöksestä, vaan perinteisten menetelmien kohdalla tehtävien ennakkopäätösten rajallisuudesta. Rajallisuus voidaan muotoilla kysymykseksi: Kuinka malli yleistyy löytämään kriittiset pisteet laajasta datasetistä rajallisella määrällä datapisteitä? 

Tähän kysymykseen syväoppivat menetelmät tarjoavat vastauksen. Nämä menetelmät oppivat datasetin tuottoprosessiin liittyviä riittävän oikeita oletuksia, jotka auttavat malleja sovittumaan tähän prosessiin rajoitetusta koulutusdatasetin koosta huolimatta. Yhdellä sanalla kysymys on datasetin rakenteesta (*structure*) ja sen oppimisesta. Perinteisten menetelmien sijaan syväoppivat menetelmät lähestyvät mallinnettavan rakenteen oppimista sillä oletuksella, että tämä rakenne muodostuu eri piirteiden yhteisvaikutuksesta (*composition of features*).

Koneoppimismenetelmien taustalla vaikuttaa vahvasti ajatus merkittävien kokonaisuuksien oppimisesta (*manifold learning*). Esimerkiksi puheentunnistuksessa ei ole mitenkään mielekästä pyrkiä oppimaan koko kuultavan äänen aluetta ja johtamaan siitä erilaisten äänteiden rakenteita, vaan puheääni keskittyy normaalisti tietyille taajuusalueille. Tällöin mielekästä on pyrkiä oppimaan näiden alueiden avulla yhtenäinen kokonaisuus, johon vähemmän merkittävät alueet vaikuttavat havaitulla variaatiolla. Näin moniulotteisen tai -piirteisen datasetin opettelu pyrkii keskittymään rajatulle alueelle ja oppiminen helpottuu. Näin myös piirteiden määrää saadaan pienennettyä, jolloin datasetin rakenteen visualisointi helpottuu.

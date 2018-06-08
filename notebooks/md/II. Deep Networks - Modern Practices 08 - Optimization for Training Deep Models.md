
*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#8.-Optimization-for-Training-Deep-Models" data-toc-modified-id="8.-Optimization-for-Training-Deep-Models-1">8. Optimization for Training Deep Models</a></span><ul class="toc-item"><li><span><a href="#8.1.-How-Learning-Differs-from-Pure-Optimization" data-toc-modified-id="8.1.-How-Learning-Differs-from-Pure-Optimization-1.1">8.1. How Learning Differs from Pure Optimization</a></span><ul class="toc-item"><li><span><a href="#Empirical-Risk-Minimization" data-toc-modified-id="Empirical-Risk-Minimization-1.1.1">Empirical Risk Minimization</a></span></li><li><span><a href="#Surrogate-Loss-Functions-and-Early-Stopping" data-toc-modified-id="Surrogate-Loss-Functions-and-Early-Stopping-1.1.2">Surrogate Loss Functions and Early Stopping</a></span></li><li><span><a href="#Batch-and-Minibatch-Algorithms" data-toc-modified-id="Batch-and-Minibatch-Algorithms-1.1.3">Batch and Minibatch Algorithms</a></span></li></ul></li><li><span><a href="#8.2.-Challenges-in-Neural-Network-Optimization" data-toc-modified-id="8.2.-Challenges-in-Neural-Network-Optimization-1.2">8.2. Challenges in Neural Network Optimization</a></span><ul class="toc-item"><li><span><a href="#Local-Minima" data-toc-modified-id="Local-Minima-1.2.1">Local Minima</a></span></li><li><span><a href="#Cliffs" data-toc-modified-id="Cliffs-1.2.2">Cliffs</a></span></li><li><span><a href="#Long-Term-Dependencies" data-toc-modified-id="Long-Term-Dependencies-1.2.3">Long-Term Dependencies</a></span></li><li><span><a href="#Inexact-Gradients" data-toc-modified-id="Inexact-Gradients-1.2.4">Inexact Gradients</a></span></li><li><span><a href="#Poor-Correspondence-between-Local-and-Global-Structure" data-toc-modified-id="Poor-Correspondence-between-Local-and-Global-Structure-1.2.5">Poor Correspondence between Local and Global Structure</a></span></li><li><span><a href="#Theoretical-Limits-of-Optimization" data-toc-modified-id="Theoretical-Limits-of-Optimization-1.2.6">Theoretical Limits of Optimization</a></span></li></ul></li><li><span><a href="#8.3.-Basic-Algorithms" data-toc-modified-id="8.3.-Basic-Algorithms-1.3">8.3. Basic Algorithms</a></span><ul class="toc-item"><li><span><a href="#Stochastic-Gradient-Descent" data-toc-modified-id="Stochastic-Gradient-Descent-1.3.1">Stochastic Gradient Descent</a></span></li><li><span><a href="#Momentum" data-toc-modified-id="Momentum-1.3.2">Momentum</a></span></li><li><span><a href="#Nesterov-Momentum" data-toc-modified-id="Nesterov-Momentum-1.3.3">Nesterov Momentum</a></span></li></ul></li><li><span><a href="#8.4.-Parameter-Initialization-Strategies" data-toc-modified-id="8.4.-Parameter-Initialization-Strategies-1.4">8.4. Parameter Initialization Strategies</a></span><ul class="toc-item"><li><span><a href="#Initialization-of-Weights" data-toc-modified-id="Initialization-of-Weights-1.4.1">Initialization of Weights</a></span></li><li><span><a href="#Initialization-of-Remaining-Parameters" data-toc-modified-id="Initialization-of-Remaining-Parameters-1.4.2">Initialization of Remaining Parameters</a></span></li><li><span><a href="#Other-Methods" data-toc-modified-id="Other-Methods-1.4.3">Other Methods</a></span></li></ul></li><li><span><a href="#8.5.-Algorithms-with-Adaptive-Learning-Rates" data-toc-modified-id="8.5.-Algorithms-with-Adaptive-Learning-Rates-1.5">8.5. Algorithms with Adaptive Learning Rates</a></span><ul class="toc-item"><li><span><a href="#AdaGrad" data-toc-modified-id="AdaGrad-1.5.1">AdaGrad</a></span></li><li><span><a href="#RMSProp" data-toc-modified-id="RMSProp-1.5.2">RMSProp</a></span></li><li><span><a href="#Adam" data-toc-modified-id="Adam-1.5.3">Adam</a></span></li></ul></li><li><span><a href="#8.6.-Approximate-Second-Order-Methods" data-toc-modified-id="8.6.-Approximate-Second-Order-Methods-1.6">8.6. Approximate Second-Order Methods</a></span><ul class="toc-item"><li><span><a href="#Newton's-Method" data-toc-modified-id="Newton's-Method-1.6.1">Newton's Method</a></span></li><li><span><a href="#Conjugate-Gradients" data-toc-modified-id="Conjugate-Gradients-1.6.2">Conjugate Gradients</a></span></li><li><span><a href="#BFGS" data-toc-modified-id="BFGS-1.6.3">BFGS</a></span></li></ul></li><li><span><a href="#8.7.-Optimization-Strategies-and-Meta-Algorithms" data-toc-modified-id="8.7.-Optimization-Strategies-and-Meta-Algorithms-1.7">8.7. Optimization Strategies and Meta-Algorithms</a></span><ul class="toc-item"><li><span><a href="#Batch-Normalization" data-toc-modified-id="Batch-Normalization-1.7.1">Batch Normalization</a></span></li><li><span><a href="#Coordinate-Descent" data-toc-modified-id="Coordinate-Descent-1.7.2">Coordinate Descent</a></span></li><li><span><a href="#Polyak-Averaging" data-toc-modified-id="Polyak-Averaging-1.7.3">Polyak Averaging</a></span></li><li><span><a href="#Supervised-Pretraining" data-toc-modified-id="Supervised-Pretraining-1.7.4">Supervised Pretraining</a></span></li><li><span><a href="#Designing-Models-to-Aid-Optimization" data-toc-modified-id="Designing-Models-to-Aid-Optimization-1.7.5">Designing Models to Aid Optimization</a></span></li><li><span><a href="#Continuation-Methods-and-Curriculum-Learning" data-toc-modified-id="Continuation-Methods-and-Curriculum-Learning-1.7.6">Continuation Methods and Curriculum Learning</a></span></li></ul></li></ul></li></ul></div>

## 8. Optimization for Training Deep Models

Koneoppimismenetelmien optimointia voidaan tehdä monella tapaa. Syväoppivien menetelmien eli neuroverkkojen kohdalla tarvitaan kuitenkin erityisesti niille kehitettyjä optimointitapoja, sillä verkon koulutus kertaalleen alusta loppuun voi viedä jopa viikkoja. Tällöin iteratiivinen optimointi on erittäin kallista. Optimoinnin tavoitteena löytää sellainen parametrijoukko $\theta$, joka minimoi kohdefunktion $J(\theta)$.

### 8.1. How Learning Differs from Pure Optimization

Syväoppivien menetelmien optimointi eroaa perinteisten menetelmien optimoinnista suorituskykymittarin eli virhefunktion kohdalla. Perinteisten menetelmien optimoinnin tavoitteena on pyrkiä minimoimaan virhefunktio itsessään. Syväoppivat menetelmät optimoidaan kuitenkin kohdefunktion osalta, jossa virhefunktio on vain yksi sen osatekijöistä. Tällöin syvien menetelmien optimointi on virhefunktioon nähden epäsuoraa.

Optimoitava kohdefunktio on usein virhefunktion $L$ keskiarvo koko koulutuksessa käytetyn datasetin kohdalla. Tällöin kohdefunktio voidaan kirjoittaa muodossa

$$ J(\theta) = \mathbb{E}_{(x,y)} L(\hat{y},y), $$

jossa $\mathbb{E}_{(x,y)}$ tarkoittaa odotusarvoa koko koulutusdatasetin suhteen. Virhefunktion parametri $\hat{y}$ voidaan myös kirjoittaa muodossa $f(x;\theta)$, mikä tarkoittaa siis mallin tuottamaa ennustetta syötteestä $x$ parametrien $\theta$ kanssa. 

Kirja käyttää myös kohdefunktion odotusarvossa seuraavia lisämerkintöjä:

- $\tilde{}\hat{p}_{data}$ tarkoittaa datasetin havaittua eli empiiristä jakaumaa.
- $\tilde{}p_{data}$ tarkoittaa datasetin tuottaneen datantuottoprosessin jakaumaa.

Tällä tahdotaan erotella koulutusdatasetin suhteen optimointi datasetin näytteet tuottaneen perustavamman prosessin suhteen optimoinnista. Tällöin tavoiteltu optimoitava kohdefunktio on 

$$ J^*(\theta) = \mathbb{E}_{(x,y)\tilde{}p_{data}} L(\hat{y},y), $$

jolla ilmaistaan odotettua yleistysvirhettä (*generalization error*).

#### Empirical Risk Minimization

Odotettu yleistysvirhe voidaan tulkita myös riskiksi, joka on laskettu datantuottoprosessin $p_{data}$ suhteen. Mikäli tämä prosessi olisi tunnettu, olisi optimointi algoritmisesti ratkaistavissa. Näin ei kuitenkaan ole, vaan tämä jakauma on pyrittävä oppimaan prosessista syntyneiden näytteiden avulla, ts. empiirisestä jakaumasta $\hat{p}_{data}$.

Tällöin ollaan tekemisissä empiirisen riskin minimoinnin kanssa (*empirical risk minimization*). Käytännössä se on virhefunktion keskiarvon laskentaa kaikkien datasetin näytteiden yli. Kun koulutettavaa mallia optimoidaan näin, toiveena on, että empiiristä riskiä eli kohdefunktiota minimoimalla minimoituu myös todellinen riski eli yleistysvirhe.

#### Surrogate Loss Functions and Early Stopping

Mikäli optimoinnin kohteena on esimerkiksi binäärinen luokkien oppiminen, ei binääriulostulojen suhteen optimointi ole välttämättä edes mahdollista. Tällöin on parempi kääntyä sijaisvirhefunktioiden (*surrogate loss function*) puoleen. Esimerkki tällaisestä on negatiivisen logaritmisen todennäköisyyden käyttö ennustetun mallin todennäköisyyden ennustamisessa, joka on myös optimoitavissa.

Näin toimittaessa voidaan hyödyntää myös aiemmin esiteltyä regularisointimenetelmää, aikaisin lopettamista (*early stopping*), ylisovittumisen ehkäisyksi. Tällöin virhefunktiota käytetään kuitenkin validointisettiin, eikä niinkään koulutussettiin.

#### Batch and Minibatch Algorithms

Koneoppimismenetelmien optimointi on edellä puhuttuun tukeutuen kohdefunktion optimointia, joka edelleen on siis datasetin näytteiden virhefunktioiden summien ja niistä lasketun keskiarvon optimointia. Kulloinkin lasketut virheet ilmentävät koulutettavan mallin kykyä mallintaa datasetin näytteet tuottanutta jakaumaan $p_{data}$.

Kuten myös on jo tuttua, koulutuksen aikana on tärkeää laskea mallin koulutettavien parametrien gradientit. Näiden laskenta on kuitenkin laskennallisesti kallista, mikäli ennen parametrien päivitystä mallin tarvitsisi nähdä datasetin kaikki näytteet.

Tällöin voidaan hyödyntää datasetistä koostettuja satunnaisotoksia. Laskemalla virheiden keskiarvot käyttäen koko datasettiin nähden huomattavasti pienempää osasettiä on gradienttien laskenta nopeampaa ja mallia voidaan päivittää myös tiheämmin. 

Satunnaisotanta takaa, että keskimääräisesti otetut näytteet ilmentävät itse datasettiä. Menetelmän käytön perusteltavuus saa myös lisätukea siitä, että datasetin kasvattaminen lisää mallin koulutettavaa tarkkuuttaa vaimenevasti, t.s. lisänäytteet lisäävät tarkkuutta vain tiettyyn pisteeseen asti. Pienemmällä datasetillä saadaan siis suurempaan nähden jo riittävästi suuntaa antava tarkkuus mutta kevyemmällä laskennalla.

Käytettyjen näytteiden osalta menetelmät voidaan jakaa seuraavasti:

- Koko datasetti (*batch*)
- Datasetin osajoukko (*minibatch*)
- Yksi näyte (*online*)

Näitä termejä käytetään kuitenkin laajalti sekaisin, joskin koneoppimismenetelmien implementaatioiden yhteydessä esiintyvä *batch size* viittaa rajoitetun satunnaisotoksen kokoon. Syväoppivat menetelmät ovat normaalisti osajoukkoja hyödyntäviä.

Käytännön näkökulmasta datasetin osittelu *minibatcheihin* vaikuttaa olennaisesti koulutuksen suorituskykyyn. Osajoukon koon ei tule olla niin pieni, ettei eimserkiksi hajautetusta laskennasta ole enää hyötyä eikä niin suuri, että muistin kanssa tulee ongelmia. Näytönohjainten kanssa on hyvä käyttää $2^n$ kokoja.

Koska *minibatch* on satunnaisotantaa takaisinpanolla, se lisää koulutukseen hieman kohinaa datantuottojakaumaan $p_{data}$ nähden. Näin se toimii myös regularisoijana, joskin suurin regularisointi saavutetaan pienimmällä mahdollisella osajoukon koolla näyte per joukko. Tämä ei ole kuitenkaan toivotuin asetelma.

Käytetyn koneoppimismallin robustisuus määrittää sopivan osajoukon koon. Gradientteihin pohjaavat menetelmät ovat robusteja, jolloin niille kelpaavat pienemmätkin joukot. Gradienttien gradientteja (Hessen matriisi) käyttävät menetelmät ovat taas herkempiä, jolloin näytteitä on oltava kerralla enemmän.

Koska osajoukkojen kanssa koulutusta tapahtuu todennäköisesti samoilla näytteillä useammin kuin kerran, opetuksesta tulee ainakin jossain määrin vääristynyttä. *Online*-tyyppisessä opettamisessa päästään kaikista lähimmäksi vääristymättömiä gradientteja, sillä silloin jokainen näyte havaitaan vain kerran. Mitä suurempia datasetit, sitä harvemmin samoja näytteitä kannattaa kierrättää.

### 8.2. Challenges in Neural Network Optimization

Neuroverkkojen kanssa ratkaistavat ongelmat ovat harvoin konvekseja, eli sellaisia, joilla on yksiselitteinen minimiarvo koulutettavien parametrien suhteen. Tästä syystä syväoppivien menetelmien optimointiin liittyy joitakin selkeitä haasteita. Kirjassa käytetään tässä alaluvussa useasti termejä *higher cost* tai *lower cost*. Näillä tarkoitetaan virhefunktion tulosta, jota pyritään minimoimaan. 

Syväoppivia menetelmiä opetettaessa lähtökohta on, että datasetin avulla tavoiteltavaan $p_{data}$ on hankala tai mahdoton mukautua (*ill-conditioned*) optimoinnin avulla. Tällöin joidenkin piirteiden minimi voi olla kuitenkin toisten maksimi (satulapiste). Tämä voidaan havaita, kun gradienttien normi $g^Tg$ ei pienene merkittävästi, kun taas $g^THg$ kasvaa huomattavasti. Tämä johtaa koulutuksen aikaseen gradienttien kasvuun, vaikka koulutuksen tarkkuus paranisikin.

#### Local Minima

Paikalliset minimiarvot (*local minima*) vaikeuttavat globaalin minimin etsintää. Neuroverkoilla on lähtökohtaisesti useita paikallisia minimejä, sillä niillä voidaan tuottaa yhtäpitäviä tuloksia kirjavilla parametrien konfiguraatioilla. Tästä syystä syviltä menetelmiltä puuttuu mallin tunnistettavuus (*model identifiability*). 

Malli on tunnistettava silloin, kun riittävän suurella koulutusdatasetillä on löydettävissä yksi parhaiten sopiva malli. Neuroverkoilla on tähän nähden vastakkainen ominaisuus, joka tunnetaan painoavaruuden symmetriana (*weight space symmetry*). Tämän lisäksi tunnistettavuutta heikentäviä tekijöitä on muitakin. Kohdefunktion minimoimisen näkökulmasta tärkeintä ei kuitenkaan ole sellaisen parametrien konfiguraation löytäminen, joka tavoittaa kohdefunktion globaalin minimin, vaan *riittävän* pienen arvon löytäminen.

Paikallisissa minimeissä gradientit saavat arvon nolla. Tämä voi tapahtua tasanteissa, satulapisteissä tai missä tahansa mallin parametriavaruuden paikassa, jossa jokin tai jotkin parametrit eivät muutu. Ongelmallisia paikalliset minimit ovat silloin, kun niiden tuottama virhearvo on globaalia minimiä suurempi. Pääteltynä tämä johtaisi ylisuuriin gradienttien korjausliikkeisiin, jolloin optimoitaisiin ohi globaalin minimin. Hyvä analogia olisi liian kova putti minigolfradalla.

*Gradient descent* on kuitenkin suunniteltu tätä ongelmaa varten. Koska se ei kelpuuta vain gradientit nollaavia pisteitä, vaan etsii aina alamäkeä, se kykenee jopa suhteellisen nopeastikin löytämään tiensä esimerkiksi satulapisteistä.

#### Cliffs

Joskus riittävän suuret painot muodostavat keskenään kerroittuna parametriavaruuden pinnassa harjanteita (*cliffs*). Nämä ovat ongelmallisia, mikäli gradientit ovat liian suuria. Jyrkkiä muutoksia on tällöin hankala seurata. Eräs ratkaisu tähän ongelmaan on käyttää gradienttien leikkausta (*gradient clipping*). Tällöin tärkeimpänä pidetään tietoa gradientin eli kulmakertoimen suunnasta ja sen magnitudia rajoitetaan. Näin muutosten askelpituuksia saadaan rajoitettua, jolloin jyrkät muutokset havaitaan helpommin.

#### Long-Term Dependencies

Mitä suurempi neuroverkko, sitä suurempia ovat sen laskentaketjut (*computational graph*). Näin on myös rekursiivisten verkkojen kanssa, jotka saavat syvyytensä ajallisessa mielessä. Laskentaketjujen kasvaessa ilmenee etenkin gradientteihin liittyviä ongelmia.

Nämä ongelmat tunnetaan paremmin joko katoavien (*vanishing*) tai räjähtävien (*exploding*) gradienttien ongelmana. Otetaan esimerkiksi rekursiivinen malli, jossa käytössä on aina sama painomatriisi $W$. Tämän matriisin arvo on $t$ rekursion jälkeen $W^t$. Gradientit skaalautuvat painojen mukaan. Mikäli gradienttien arvot ovat tällöin jotain muuta kuin 1, vaimenevat tai räjähtävät gradientit.

Perinteisemmät myötäkytketyt verkot kuitenkin välttävät tätä ongelmaa melko tehokkaasti, sillä peräkkäiset neuronikerrokset ovat arvoiltaan identtisiä hyvin pienellä todennäköisyydellä mm. painojen satunnaisen alustamisen ja mahdollisten topologisten erojen vuoksi.

#### Inexact Gradients

Koska mallia opetetaan vain datantuottoprosessin näytteillä, eivät lasketut gradientit ole täydellisiä. Niissä on usein kohinaa ja ne ovat jossain määrin vääristyneitä. Samoin mikäli kohdefunktio on hankalasti optimoitava, ovat myös gradientit samaan tapaan käyttäytyviä. Tällöin sijaiskohdefunktion oikea valinta voi auttaa.

#### Poor Correspondence between Local and Global Structure

Käytännössä neuroverkot eivät juuri koskaan asetu yhteenkään kriittiseen pisteeseen. Koulutuksen edetessä ja mallin tullessa varmemmaksi lähestyy jotain tällaista pistettä asymptoottisesti eli lähestyen, mutta saavuttamatta. Tällöin paikallista minimiä tai harjannettakin suuremmaksi muodostuu oikeaan suuntaan tapahtuva optimointi. Väärään suuntaan optimoimatessa paikallisen ongelmakohdan selvittäminen viekin kauemmas kauempana siintävästä globaalin minimin tai edes merkittävästi pienemmän virheen alueesta.

Tällöin minimoinnin sijasta tärkeämmäksi muodostuu lyhimmän reitin löytäminen kohti matalamman virheen alueita. Tähän *gradient descent* ei enää suoraan auta, sillä se keskittyy etenkin paikallisiin muutoksiin. Tämä on jatkuvan tutkimuksen aluetta, eikä kirjoittajillakaan ole kuin ehdotuksia tutkimussuuntien suhteen.

#### Theoretical Limits of Optimization

Vaikka teoreettisesti on voitu osoittaa, että millä tahansa optimointialgoritmilla on ongelmia neuroverkkojen kanssa, ei käytännössä tällä ole kovin suurta vaikutusta. Teoreettiset tarkastelut nojaavat diskreetteihin arvoihin neuronien ulostuloina, mutta jo aiemminkin käsiteltyyn pohjaten neuroverkot kykenevät tekemään paikallista optimointia pienin iteratiivisin muutoksin. Neuroverkkojen optimoinnin teoreettinen analyysi on erittäin vaativaa.

### 8.3. Basic Algorithms

Tässä alaluvussa esitellään *gradient descentin* laajennos *stochastic gradient descent* eli SGD, joka kiihdyttää muutoin jopa verkkaista algoritmia satunnaisilla minibatcheillä.

#### Stochastic Gradient Descent

Gradienteista on mahdollista saada vääristymätön arvioi (*estimate*) käyttämällä $p_{data}$-jakaumasta koostetun minibatchin gradienttien keskiarvoa. Gradienttien keskiarvo painotetaan oppimiskertoimella $\epsilon$, minkä jälkeen mallin parametrit päivitetään vähentämällä parametrien arvoista oppimiskertoimella painotetut gradienttien keskiarvot.

SGD:n osalta oppimiskerroin on tärkeässä roolissa. Oikean oppimiskertoimen valinta on pitkälti yrityksen ja erehdyksen kautta tapahtuva iteratiivinen prosessi. Valitun oppimiskertoimen vaikutuksia voidaan arvioida piirtämällä kohdefunktion kehitystä koulutuksen aikana ja vertailemalla tuotettuja kuvaajia eri oppimiskertoimien kesken. 

Vakiona pysyvän arvon sijasta sitä voidaan myös muuttaa koulutuksen aikana, joka käytännössä tarkoittaa oppimiskertoimen pienentämistä asteittain. Sitä ei kuitenkaan voida kokonaan häivyttää, sillä muutoin malli lakkaisi oppimasta. Tällöin sitä pienennetään vain tiettyyn rajaan asti. Lineaarisesti pienenevä oppimiskerroin voidaan tällöin ilmaista kaavalla

$$ \epsilon_k = \max{\left((1 - \alpha)\epsilon_0, 0 \right)} + \alpha\epsilon_{\tau},$$

jossa $\alpha = \frac{k}{\tau}$, $k$ on sen hetkinen iteraatio ja $\tau$ on iteraatioiden raja. 

Tavallisesti raja asetetaan muutamaan sataan koulutuskierrokseen. Samoin $\epsilon_{\tau}$ asetetaan tyypillisesti tasoon $\tilde{}0.01\epsilon_0$.Oppimiskertoinen alkuarvon $\epsilon_0$ on haastavaa. Tyypillistä on, että se on jonkin verran suurempi kuin oppimiskerroin, joka antaa parhaan tuloksen n. 100 koulutuskierroksen jälkeen.

Yksi SGD:n hyödyllisistä ominaisuuksista on sen kyky supistua. Vaikka koulutusdatasettiä kasvatettaisiin, minibatchien käytön vuoksi koulutusvirhe voidaan saada riittävän pieneksi jo ennen kuin kaikkia näytteitä on ehditty käsitellä. Vaikka koko koulutussettiä käyttämällä eli batch-koulutuksella mallin testivirhe voidaan saada pienemmäksi, ei ero minibatch-koulutukseen ole merkittävä.

####  Momentum

Jopa SGD voi olla ajoittain hidas, mikäli parametriavaruudessa on paljon harjanteita tai gradientit ovat joko hyvin kohinaisia tai pieniä ja yhtenäisiä. Tällöin koulutukseen voidaan lisätä koulutuksen mukana muuttuva liukuva keskiarvo edeltäneistä gradienteista, jolloin koulutuksella on sananmukaisesti liike-energiaa eli tapahtuneen koulutuksen suuntaan.

Tällöin mallin parametreja ei päivitetä vain kulloinkin laskettujen gradienttien mukaisesti, vaan gradienttien $g$ laskennan jälkeen päivitetään mallin suunta- ja nopeusvektori $v$, jonka avulla sitten mallin parametrit $\theta$ päivitetään. Tällöin

$$ \theta = \theta + v = \theta + (\alpha v - \epsilon g), $$

jossa $\alpha$ on jälleen momentin painotuskerroin ja $\epsilon$ on oppimiskerroin. Kertoimien keskenään suhteelliset suuruudet määrittävät, missä määrin uusimpia gradientteja painotetaan suhteessa menneistä laskettuun momenttiin.

Tällä on se suora vaikutus, että enää gradientteja ei päivitetä sokeasti vain viimeisimmän gradientin suunnan ja suuruuden mukaan. Mikäli uusi gradientti $g$ ei olekaan enää samansuuntainen momenttivektorin $v$ kanssa, vaimenee $v$ kertoimien mukaisesti ja vaihtaa suuntaansa myös hieman. Kuten oppimiskerrointakin, voidaan momenttiakin säätää oppimisen aikana. Se ei kuitenkaan ole yhtä merkityksekästä, kuin oppimiskertoimen koulutuksenaikainen säätäminen. 

Käytännössä momentin käyttö vaimentaa koulutusten välistä vaihtelua ja auttaa gradienttia etenemään selkeämmässä linjassa. Näin se myös auttaa helposti poukkoilevaa gradienttia asettumaan.

#### Nesterov Momentum

Kuten monista koneoppimisen menetelmistä, myös momentista löytyy variaatioita. Eräs tällainen on Nesterovin momentti, joka eroaa edellä esitellystä momentista gradienttien laskennan hetken suhteen. Tavallisessa momentissa $g$ lasketaan ensin, sitten päivitetään $v$ ja viimeiseksi vasta $\theta$. Nesterovin malli pyrkii käyttämään $v$:tä ennemmin korjausmuuttujana, laskien gradientit $g$ momentin huomoivilla parametreilla, minkä jälkeen päivitetään momentti ja parametrit.

### 8.4. Parameter Initialization Strategies

Toisin kuin perinteisten menetelmien kanssa, syväoppivien menetelmien parametrit on alustettava jossain määrin manuaalisesti. Tämä on ongelmallista etenkin, kun mallin sovittuvuus voi riippua koulutuksen aloituspisteestä. Jotkin alkuasetelmat voivat olla niin epästabiileja, että malli ei koulutu lainkaan. Toisissa tapauksissa malli kyllä voi kouluttua, mutta hankalan aloituspisteen vuoksi se voi tapahtua huomattavan hitaasti.

Nykyaikaiset alustusmenetelmät ovat pääsääntöisesti simppeleitä ja heuristisia. Koska neuroverkkojen optimointi on toistaiseksi vasta tutkimuksen kohteena ilman syvää saavutettua ymmärrystä, alustusstrategiat ovat ennemmin empiirisesti vahvistettavia kuin teoreettisesti perusteltuja. Asiaa ei yhtään helpota, että optimoinnin näkökulmasta parhaimmat parametrien alkuarvot voivat olla yleistyvyyden kannalta huonoja.

#### Initialization of Weights

Tärkeimpänä tunnettuna alustuksen ominaisuutena pidetään neuronien välisen symmetrian rikkoutumista. Tällöin saman aktivointifunktion omaavilla ja samoihin edeltäviin neuroneihin yhdistetyillä neuroneilla tulee olla eri alkuarvo. Vain näin ne alkavat eriytyä koulutuksen edetessä. 

Tyypillisesti tämä tarkoittaa riittävän pieniä satunnaisia alkuarvoja. Samoin on tyypillistä, että satunnaisalustus koskee vain mallin painoja, eikä niinkään esimerkiksi *bias*-yksikköjä. Mikäli arvot alustetaan liian suuriksi, voivat lasketut väliarvot räjähtää liian suuriksi. Tämä on ongelma esimerkiksi aktivointifunktioiden saturoituvuuden näkökulmasta. Samoin mallista voi kaoottisen epästabiili. Optimoinnin ja regularisoinnin tavoitteet ovat alustuksen osalta vastakkaisia. Optimointi tapahtuu varmemmin suurilla painoilla, kun taas regularisointi hyötyy pienemmistä painoista.

Alustusheuristiikkoja on monia. Normalisoitu alustus (*normalized initialization*) pyrkii löytämään tasapainon tasaisen aktivointi- ja gradienttivarianssin välillä käyttämällä syöte- ja ulostuloyksikköjen määrää määrittävänä tekijänä. Toisaalta pelkkää syötteidenkin määrää voidaan käyttää. Toisaalta alustus voidaan tehdä myös käyttämällä vastakohtaisia skaalattuja matriiseja epälineaarisuuksien takaamiseksi. 

Nämä eivät kuitenkaan takaa optimaalisinta suorituskykyä. Kaikki alustusstrategiat eivät sovellu kaikille malleille eivätkä mallinnettaville ongelmille. Koulutus voi myös hävittää alkuperäisen aiotun jakauman. Optimoinnin nopeutus ei myöskään takaa parempaa yleistyvyyttä. Jakaumaan pohjaavat alustusmenetelmät voivat johtaa myös tarpeettoman pieniin aloituspainoihin.

Eräs vaihtoehto viimeistä kohtaa ajatellen on harva alustus (*sparse initialization*). Tällöin jokaiselle neuronille sallitaan vain tietty määrä nollasta poikkeavia painoja. Tämäkään ei kuitenkaan ole ongelmaton lähestymistapa, sillä se pakotta vain vahvemman valitun jakauman mukaisen käyttäytymisen painoissa.

Laskentakapasiteetin salliessa painojen alustuksen satunnaisuuden variaation skaalaa kannattaa lähestyä hyperparametrina. Tällöin kuhunkin tilanteeseen sopiva alustus määritetään usean alkuarvon välillä. Optimaalisimpia arvoja voidaan selvittää myös manuaalisesti yhdellä minibatchilla. Silloin kannattaa tarkastella aktivointifunktioiden varianssia ja kasvattaa painoja, mikäli varianssi pienenee liian nopeasti liian pieneksi. Kun tämä toteutetaan vielä kerros kerrallaan, saadaan melko helposti selville sopivat painot koko verkolle.

#### Initialization of Remaining Parameters

Muiden parametrien alustus on jo helpompaa. Näitä tyypillisesti ovat mallin kerros- tai neuronikohtaiset *bias*-yksiköt. Ulostulokerroksessa *bias*-yksikkö voidaan säätää tuottamaan mahdollisimman harhattomia tuloksia. Piilokerroksissa samaisten yksikköjen käyttö voi olla perusteltua koulutuksen alussa tapahtuvan ennenaikaisen saturaation välttämiseksi. Joskus taas on merkityksellistä, jokin aktivointifunktio on lähtökohtaisesti aktiivinen. Tällöin *bias* voidaan säätää joksikin aktivoinnin takaavaksi arvoksi.

#### Other Methods

Alustus voidaan toteuttaa myös koneoppimismenetelmin. Silloin käytetään ohjaamatonta mallia tuottamaan parametreille alkuarvot, joita sitten hyödynnetään varsinaisessa mallissa.

### 8.5. Algorithms with Adaptive Learning Rates

Tässä aliluvussa keskitytään mukautuviin *minibatch*-optimointialgoritmeihin. Oppimikerroin on eräs vaikeiten säädettävä hyperparametri, sillä sen vaikutus mallin suorituskykyyn on erityisen merkittävä. Momentin, eli yhden ylimääräisen hyperparametrin, käyttämisen sijasta oppimiskertoimesta voidaan tehdä myös koulutuksen aikana mukautuva (*adaptive*). 

Toisaalta adaptiiviset menetelmät hyödyntävät myös momentteja ja joskus esittelevät omia hyperparametrejaan. Konsensusta siitä, mikä menetelmä tulisi yleisesti valita, ei kuitenkaan ole. Vaikkakin mukautuvan oppimiskertoimen optimointialgoritmit ovat osoittautuneet muita optimointialgoritmeja hieman paremmiksi, ei niiden keskuudesta ole löytynyt toistaiseksi selkeästi muita parempaa vaihtoehtoa. Suosituimmat ovat *SGD*, *RMSProp*, *AdaDelta* ja *Adam* - momentilla ja ilman.

#### AdaGrad

Tämä algoritmi skaalaa parametrikohtaiset oppimiskertoimet käänteisesti menneisiin gradientteihin nähden. Näin koulutuksen aikana suurimman kohdefunktion osaderivaatan eli gradientin omaavan parametrin oppimiskerrointa pienennetään tällöin eniten. Menetelmä ei ole kuitenkaan ongelmaton, sillä gradienttien kuljetus alusta asti voi aiheuttaa liian suuren gradienttien leikkauksen ennenaikaisesti.

AdaGrad toimii parhaiten konvekseissa eli kuperan parametriavaruuden ongelmissa, mutta kärsii monia lokaaleja minimejä sisältävissä parametriavaruuden pinnoissa. Koska oppimiskerrointa pienennetään koko ajan, ei algoritmi välttämättä pääse lokaalista minimistä enää pois.

#### RMSProp

Algoritmi on AdaGradin jatkokehityksen tulosta, jossa gradienttin alusta asti jatkuvan keruun sijaan menneitä gradientteja lasketaan liukuvan ikkunan sisällä. Näin menetelmä kykenee paremmin etenemään lokaaleista minimeistä kohti matalamman kohdefunktion arvon kohtaa. Samoin matalan kohdan löydyttyä mallin koulutus valmistuu nopeasti. Sitä voidaan käyttää tehokkaasti myös yhdessä momentin kanssa. 

Käytännössä menetelmä on osoittautunut luotettavaksi ja se onkin laajalti käytetty syväoppivien menetelmien koulutuksessa. Menetelmä tarvitsee kuitenkin toimiakseen yhden lisähyperparametrin $\rho$, joka määrittää liukuvan ikkunan koon.

#### Adam

Jos RMSProp laajensi AdaGradia gradienttien laskennan suhteen, Adam (*adaptive moments*) laajentaa RMSPropia etenkin momentin osalta. Momentin käyttö on sisäänrakennettu algoritmiin siten, että momentti olisi koulutuksen alusta alkaen mahdollisimman vääristymätön. Adam on myös robustimpi hyperparametrien valinnan suhteen.

### 8.6. Approximate Second-Order Methods

Tässä luvussa keskitytään toisen asteen gradientin optimointimenetelmiin. Esimerkkinä käytetään empiiristä riskiä, joka toisin sanoen tarkoittaa koulutusdatasta laskettua näytteiden keskivirhettä. Menetelmiä voidaan kuitenkin myös laajentaa sisältämään regularisointimenetelmiä ja muita aiemmin keskusteltuja kohdefunktioon yhdistettäviä komponentteja.

#### Newton's Method

Tämä menetelmä on kaikkein käytetyin toisen asteen eli gradienttien gradientteihin pohjaava optimointimenetelmä. Menetelmä pyrkii approksimoimaan kohdefunktion minivoivan parametrijoukon $\theta^*$ hyödyntämällä Hessen eli toisen asteen gradienttien matriisia $H$, etenkin sen inverssiä eli käänteismatriisia $H^{-1}$. 

Tämän matriisin $H$, eli gradienttien gradienttien, on kuitenkin oltava täysin positiivisia lukuja täynnä, sillä muutoin menetelmä ei toimi oikein. Jos $H$ ei ole täysin positiivinen, esimerkiksi parametriavaruuden satulapisteeseen joutuessaan menetelmä voi optimoida mallin parametreja väärään suuntaan. Tällöin ja samoin siis siksi tarvitaan $H$:n regularisointia.

Pelkän skalaarin $\alpha$ lisääminen $H$:n diagonaaliin auttaa jo alkuun. Tällöin negatiivisten $H$:n arvojen on kuitenkin oltava vielä riittävän lähellä nollaa. Mitä negatiivisempia arvoja matriisista löytyy, sitä suurempaa $\alpha$:n arvoa on käytettävä. Tällöin kuitenkin diagonaali alkaa ylikorostua ja sen ulkopuolella olevien arvojen vaikutus vaimenee.

Menetelmän käyttö on laskennallisesti kuitenkin erityisen raskas syväoppivien menetelmien kanssa. Tähän on syynä yksinkertaistenkin mallien käsiin räjähtävä parametrimäärä $n_p$, joille laskettava matriisi $H$ omaa $n_p^2$ alkiota.

#### Conjugate Gradients

Yhdistelmägradienttien (*conjugate gradients*) menetelmä on Newtonin menetelmää tehokkaampi, sillä se välttää $H^{-1}$ laskentaa pienentämällä kohdefunktiota iteratiivisesti gradienttien yhdistelmän suuntaan. Menetelmä pohjaa aiemmin esitellyn perinteisen jyrkimmän laskun metodin tehottomuuteen keskittyneeseen tutkimukseen. Perinteisesti tämä menetelmä toimii etenkin lineaaristen ongelmien kanssa.

Tietyissä tapauksissa jyrkimmän laskun menetelmä kykenee pienentämään kohdefunktiota vain luovimalla ees-taas. Näin se tekee turhaksi aina edellisen optimointiaskeleen päättelemän suunnan. Yhdistelmägradienttien menetelmä pyrkii tämän vuoksi hyödyntämään kulloista optimointiaskelta edeltänyttä suuntaa ikäänkuin vektorien summina. Se toimii silti jyrkimmän laskun tavoin, etsien optimoinnin rataa kohti pienempää kohdefunktion arvoa.

Käytännössä menetelmä toimii kuin momentti. Edellisen askeleen suunnan hyödyntämisen magnitudia säädellään parametrilla $\beta$, jonka laskentaan on joitakin $H$:n ominaisvektoreita hyödyntäviä menetelmiä. Joka tapauksessa parametrin laskentavalla varmistetaan, ettei edellinen suunta dominoi uusimman optimointiaskeleen suuntaa. 

Menetelmän on kuitenkin hyödynnettävissä myös syväoppivien menetelmien kanssa joillakin muutoksilla. Toimiakseen epälineaarisissa eli monia kriittisiä pisteitä omaavissa ongelmissa, menetelmä on aika ajoin alustettava riittävän monen suunnan tutkimiseksi. Toisaalta menetelmän käyttöä voidaan myös pohjustaa SGD:n avulla, minkä jälkeen oikean parametriavaruuden lähistön löydyttyä voidaan siirtyä yhdistelmägradienttien menetelmän käyttöön.

#### BFGS

Viimeinen kirjan käsittelemistä toisen asteen gradienttien optimointimenetelmistä on Broyden-Fletchen-Goldfarb-Shanno- eli BFGS-algoritmi. Vaikkakin se on samankaltainen yhdistelmägradienttien menetelmän kanssa, se pyrkii approksimoimaan Newtonin menetelmän vaatiman laskennallisesti resurssi-intensiivisen matriisin $H^{-1}$ toisella matriisilla $M_t$. Tämä matriisi tarkentuu iteratiivisesti.

Yhdistelmägradienttien sijasta BFGS ei ole kovin riippuvainen optimaalisimpien minimointipolkujen löytämisestä. Tällöin sen iteraatiot ovat nopeampia. Sen kuitenkin on ylläpidettävä Hessen matriisin inverssin approksimaatiota $M_t$, mikä kasvattaa muistin resurssivaatimuksia. Tämä asettaa suurimmat rajoitteet sen käytettävyydelle syväoppivien menetelmien kanssa. Menetelmästä on myös rajoitetun muistin (*limited memory*) versio. Tällöin pyritään koko approksimaation sijasta tallettamaan vain edellinen approksimaatio $M_{t-1}$, jota käsitellään kyseisen optimointiaskeleen identiteettimatriisina.

### 8.7. Optimization Strategies and Meta-Algorithms

Algoritmien lisäksi on muitakin tapoja optimoida. Nämä ovat ennemmin toimintatapoja tai strategioita. Ne käsitellään tässä luvussa.

#### Batch Normalization

Osajoukon normalisointi (*batch normalization*) on yksi viimeaikisimmista kehityksistä syväoppimisen optimoinnin alueella. Algoritmin sijasta se on ennemmin mukautuvaa parametrien uudelleen määrittämistä eli reparametrisointia, jonka kehitys juontaa erityisen syvien mallien koulutuksen hankaluuteen.

Mallin parametrien päivittäminen gradienttien avulla on käytännön tasolla hieman vääristynyt lähtökohta. Pohja-ajatuksena on laskea kullekin parametrille sen muutos sillä oletuksella, että muut parametrit eivät muutu. Näin ei kuitenkaan käytännössä ole, sillä mallin kaikki parametrit päivitetään kerralla ja näin mallin kerrokset voivat muuttua suhteessa toisiinsa jopa radikaalistikin.

Osajoukkojen normalisointi pyrkii vaikuttamaan tähän koulutuksen aikaisesti. Olkoot $\theta_l$ mallin yhden kerroksen parametrijoukko ja $x_l$ siihen tulevat syötteet (joko syöte- tai piilokerroksesta). Tavoitteena on tällöin normalisoida syötteet mallin parametrien mukaan siten, että normalisoitu syötejoukko 

$$ x' = \frac{x-\mu_{\theta_l}}{\sigma_{\theta_l}}. $$

Toisin sanoen syötteet normalisoidaan suhteessa kerroksen parametreihin.

Tietyssä mielessä tämä normalisointi muodostaa myös oman piilokerroksensa, sillä gradientteja laskiessa myös normalisointi otetaan sille kuuluvassa vaiheessa huomioon. Näin on myös käytännössä, sillä esimerkiksi `PyTorch`-kirjastossa `BatchNormalization` on oma kerroksensa omine painoinensa ja bias-yksikköinensä. Näin myös tapahtuu aiemmin mainittu mallin reparametrisointi.

Normalisoinnin ottaminen osaksi gradientin laskentaa oli menetelmän merkittävin innovaatio. Ennen kaikkea se pyrkii palauttamaan syötteen edeltävää jakaumaa. Lineaaristen mallien kohdalla tästä seuraisi, että normalisoinnin seurauksena myös seuraavien kerrosten ulostulot ovat pääsääntöisesti samasta jakaumasta. Tämä keventäisi mallin päivittämisen laskennallista taakkaa tehden siitä helpomman koulutettavan, mutta samanaikaisesti se tekisi myös normalisointia seuraavat lineaariset kerrokset turhiksi.

Syvien menetelmien kanssa ongelmaa ei kuitenkaan synny, sillä ne ovat epälineaarisia ja lineaarisus täten katkeaa kerrosten välillä. Epälineaaristen mallien kohdalla menetelmän hyöty tulee siinä, että se standardisoi mallin yksiköiden parametrien keskiarvon ja keskihajonnan, jolloin oppimisesta tulee stabiilimpaa. Samanaikaisesti mallin parametrien suhteelliset suuruudet säilyvät.

Koska kyseessä on käytännössä uuden kerroksen lisääminen, on sijoittelulla myös merkitystä. Suosituksena on, että normalisointi tapahtuisi kerroksessa ennen aktivointifunktiota. Tällöin on myös tärkeää, että kerroksen mahdollinen *bias*-yksikkö kytketään pois käytöstä, sillä se muuttuu käytännössä turhaksi ja vain vaikeuttaa oppimista. 

#### Coordinate Descent

Joskus optimointi on ratkaistavissa nopeasti, kun optimoitava ongelma ositellaan pienemmiksi kokonaisuuksiksi. Piirre kerrallaan optimoinen saavuttaa varmasti edes paikallisen minimin. Kun yksittäisten piirteiden sijasta optimoidaan piirteet osajoukko kerrallaan, ollaan tekemisissä koordinaattijoukon minimoinnin (*block coordinate descent*) kanssa.

Järkevintä menetelmän käyttö on, kun syötteen piirteet voidaan ositella joukon sisällä samankaltaisiin ja joukkoina selkeästi eroaviin kokonaisuuksiin. Samoin erotteluperuste voi olla myös optimoinnin tehokkuuden erot. Mikäli piirteet ovat kuitenkin riippuvaisia toisistaan, ei menetelmän hyödyntäminen ole perusteltua.

#### Polyak Averaging

Syötteiden lisäksi myös itse gradientteja voidaan joukottaa. Näin toimitaan Polyakin keskiarvomenetelmässä. Menetelmä laskee yksinkertaisuudessaan keskiarvon muutamasta *gradient descentin* löytämästä pisteestä esimerkiksi liukuvan ikkunan mukaan. Tavoitteena on tasoittaa *gradient descentin* etenemistä hieman momentin tapaan. Suunnan sijasta lasketaan kuitenkin keskiarvo.

#### Supervised Pretraining

Koulutusta voidaan optimoida myös osittelemalla itse opettava malli. Kompleksin mallin alusta loppuun kouluttamisen sijaan voidaan ensin kouluttaa yksinkertaisempi malli, mitä sitten tehdään asteittain kompleksisemmaksi. Tällöin myös ongelma ositellaan useammaksi yksinkertaisemmaksi ongelmaksi. Tätä kutsutaan esikoulutukseksi (*pretraining*). Kun kyseessä on ohjatun oppimisen ongelman oppiminen pienemmissä osissa, on kyseessä ahne ohjattu esikoulutus (*greedy supervised pretraining*).

Ahneeksi esikoulutuksen tekee se, että osaongelmiin optimoidut mallit optimoituvat myös vain osaan suuremmasta ongelmasta. Näin näiden osamallien yhdistäminen ei automaattisesti takaa, että yhdistelmämallin (*joint model*) suorituskyky on optimaalinen, joskin se on usein ainakin riittävän hyvä.

Esikoulutusstrategioita on useita. Se voidaan toteuttaa kerros kerrallaan, jolloin kerrokset koulutetaan osana yksinkertaista mallia ja lisätään valmiina osaksi isompaa kokonaismallia. Toisaalta se voidaan toteuttaa myös siten, että ensin koulutetaan pienempi malli, jonka osia sitten hyödynnetään syvemmän mallin koulutuksen nopeuttamiseksi. Toisaalta myös pienempiä osamalleja voidaan käyttää sellaisenaan tuottamaan syötteet laajemman mallin yksittäisiin kerroksiin.

#### Designing Models to Aid Optimization

Optimointialgoritmin kehitys ei aina paranna optimointia syvien menetelmien kohdalla, vaan useammin mallin suunnittelu lähtökohtaisesti optimoitavammaksi. Aktivointifunktioiden, optimointialgoritmien jne. valinta tulisi tapahtua ennemmin optimoitavuuden kuin tehokkuuden perusteella. Samoin arkkitehtuuriset valinnat, kuten osakytketyt vs. täysin kytketyt kerrokset vaikuttavat mallin optimoitavuuteen jo edes laskennallisten vaatimusten näkökulmasta.

#### Continuation Methods and Curriculum Learning

Viimeisenä ei-algoritmisena optimoinnin menetelmänä esitellään jatkuvuusmenetelmät (*continuation methods*). Näiden tavoitteena on alustaa mallin parametrit optimaalisesti helpomman minimointipolun löytämiseksi. Menetelmät asettavat mallille joukon toistaan vaativammin optimoitavia kohdefunktioita, joissa edeltävän kohdefunktion hyvä ratkaisu on kelvollinen seuraavan aloituspiste. Nämä menetelmät ovat olleet erityisen hyviä viimeaikaisissa tutkimuksissa esim. tekoälyn alueella.

Jatkuvuusmenetelmät pyrkivät tietyssä mielessä sumentamaan parametriavaruuden pintaa, jolloin siitä voi helpommin löytyä konveksisuutta. Tällä pyritään toisin sanoen estämään mallin jumittumista paikalliseen minimiin. Koska nykyisin ollaan kuitenkin sitä mieltäl, etteivät paikalliset minimit ole suurin yksittäinen este syvien mallien yleistyvyydelle, jatkuvuusmenetelmiä hyödynnetään ennemmin koulutuksen optimoinnissa.

Koneoppimisessa menetelmää hyödynnetään opetusohjelmallisena oppimisena (*curriculum learning*), jossa aikaisempien kohdefunktioiden ratkaisu helpottuu yksinkertaistemmin opittavien näytteiden painottamisella. Sitä on hyödynnetty onnistuneesti mm. tekstin- ja kuvantunnistuksen alueilla ja se nojaa vahvasti myös ihmisten tapaan oppia ja opettaa. Samoin rekursiiviset verkot ovat hyötyneet siitä, että niille syötetään koulutuksen yhteydessä myös aina jokin satunnainen joukko helposti opittavia näytteitä. 
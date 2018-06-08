
*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#19.-Approximate-Inference" data-toc-modified-id="19.-Approximate-Inference-1">19. Approximate Inference</a></span><ul class="toc-item"><li><span><a href="#19.1-Inference-as-Optimization" data-toc-modified-id="19.1-Inference-as-Optimization-1.1">19.1 Inference as Optimization</a></span></li><li><span><a href="#19.2-Expectation-Maximization" data-toc-modified-id="19.2-Expectation-Maximization-1.2">19.2 Expectation Maximization</a></span></li><li><span><a href="#19.3-MAP-Inference-and-Sparse-Coding" data-toc-modified-id="19.3-MAP-Inference-and-Sparse-Coding-1.3">19.3 MAP Inference and Sparse Coding</a></span></li><li><span><a href="#19.4-Variational-Inference-and-Learning" data-toc-modified-id="19.4-Variational-Inference-and-Learning-1.4">19.4 Variational Inference and Learning</a></span><ul class="toc-item"><li><span><a href="#Discrete-Latent-Variables" data-toc-modified-id="Discrete-Latent-Variables-1.4.1">Discrete Latent Variables</a></span></li><li><span><a href="#Calculus-of-Variations" data-toc-modified-id="Calculus-of-Variations-1.4.2">Calculus of Variations</a></span></li><li><span><a href="#Continuous-Latent-Variables" data-toc-modified-id="Continuous-Latent-Variables-1.4.3">Continuous Latent Variables</a></span></li><li><span><a href="#Interactions-between-Learning-and-Inference" data-toc-modified-id="Interactions-between-Learning-and-Inference-1.4.4">Interactions between Learning and Inference</a></span></li></ul></li><li><span><a href="#19.5-Learned-Approximate-Inference" data-toc-modified-id="19.5-Learned-Approximate-Inference-1.5">19.5 Learned Approximate Inference</a></span><ul class="toc-item"><li><span><a href="#Wake-Sleep" data-toc-modified-id="Wake-Sleep-1.5.1">Wake-Sleep</a></span></li></ul></li></ul></li></ul></div>

## 19. Approximate Inference

Monien todennäköisyysmallien koulutus on hankalaa, sillä piilokerroksia sisältävien graafimaisten mallien tarkka inferenssi on usein lähes mahdotonta laskea kaikkien mahdollisten muuttujien jakaumien laskennan resurssi-intensiivisyyden vuoksi. Tässä luvussa käsitellään joitakin tekniikoita tämän ongelman ratkaisemiseksi. Inferenssillä tarkoitetaan muuttujien todennäköisyysjakauman laskentaa suhtessa toisiin muuttujiin.

### 19.1 Inference as Optimization

Inferenssi voidaan nähdä optimointiongelmana, jolloin likimääräisten inferenssialgoritmien johtaminen mahdollistuu. Lähestymistapana on tällöin inferenssin optimoinnin likimääräistäminen. Mikäli piilomuuttujia sisältävän mallin koko todennäköisyysjakauman $p(x;\theta)$ laskenta on haastavaa, voidaan aloittaa laskemalla todennäköisyyksien alarajat. Tämä tunnetaan mm. vaihtelevana vapaana energiana (*variational free energy*). Tekstissä puhutaan mallin yksikköjen osalta näkyvistä muuttujista $v$ ja piilomuuttujista $h$, mutta selkeyden vuoksi näkyvät muuttujat rinnastetaan itse näytteiden muuttujiin $x$.

Vaihteleva vapaa energia $\mathcal{L}(x,\theta,q)$ lasketaan näytteiden muuttujien ja mallin parametrien lisäksi piilomuuttujien jakauman $q$ avulla. Riippuen $q$:n jakaumasta, $\mathcal{L}$ voi olla alkuperäistä jakaumaa huomattavasti helpompi laskea. Sen tuottama todennäköisyys on maksimissaan sama, kuin tavoitellulla todennäköisyysjakaumalla $p(x;\theta)$. Mikäli $q$ vastaa täysin jakaumaa $p(h \mid x)$, niin $\mathcal{L}(x,\theta,q) = p(x;\theta)$.  Tällöin inferenssialgoritmin optimointi muotoutuu sellaisen $q$:n etsinnäksi, joka maksimoi $\mathcal{L}$:n eli tuo sen mahdollisimman lähelle $p$:tä.

### 19.2 Expectation Maximization

Ensimmäinen esitelty vaihtelevan vapaan energian eli $\mathcal{L}$:n maksimointiin keskittyvä algoritmi on odotusarvon maksimoinnin (*expectation maximization*) algoritmi, mikä on yleinen piilomuuttujia sisältävien mallien kanssa. Algoritmi iteratiivinen ja kaksijaksoinen. Ensimmäisessä jaksossa (E-askel) lasketaan näyte- tai osajoukkokohtaiset piilomuuttujien jakaumat $q$, minkä jälkeen maksimoidaan joko $\mathcal{L}$ joko osin näytekohtaisesti tai kokonaan erillisellä optimointialgoritmilla (M-askel). Tämä on käytännössä stokastiseen gradienttien nousuun perustuva algoritmi joko näytekohtaisesti, osajoukottain tai koko näytejoukolla.

Ensimmäisessä vaiheessa maksimoidaan siis $\mathcal{L}$ suhteessa $q$:hun (inferenssi), toisessa vaiheessa taas $\mathcal{L}$ suhteessa $\theta$:an (oppiminen). Menetelmä mahdollistaa suurten oppimisaskelien ottamisen kiinteän $q$:n kanssa.

### 19.3 MAP Inference and Sparse Coding

Piilomuuttujamallien kanssa kiinnostavin inferenssi on usein $p(h \mid x)$. Vaihtoehtoisesti voidaan laskea myös yksi kaikkein todennäköisin arvo piilomuuttujille kokonaisen jakauman sijasta. Tällöin laskennan kohteena on selvittää $h^* = \arg \max p(h \mid x)$, mikä tunnetaan maksimaalisena jälkikäteisinferenssinä (*maximum a posteriori, MAP*). Laskenta ei ole kuitenkaan likimääräisen arvon laskentaa, vaan tarkkaa maksimoivien arvojen selvitystä.

Optimoinnin suhteen MAP-algoritmi voidaan kuitenkin nähdä $q$ likimääräisenä optimointina, sillä algoritmia voidaan käyttää odotusarvon maksimoinnin ensimmäisessä askeleessa piilomuuttujien arvojen arvioinnissa. Menetelmää käytetään etenkin harvan koodauksen yhteydessä, mikä on siis lineaarinen harvuutta mallin parametrien arvoihin tuottava faktorimalli.

### 19.4 Variational Inference and Learning

Variaatio-oppimisen (*variational learning*) pohjana on ajatus vaihtelevan vapaan energian $\mathcal{L}$ maksimoinnista rajoitetulla jakaumien $q$ joukolla, joilla varsinaisen jakauman $p$ odotusarvon laskenta on kepeää. Tyypillisesti kepeyttä tavoitellaan asettamalla ositussääntöjä $q$:lle. Keskiarvokentän (*mean field*) lähestymistavalla 

$$ q(h \mid x) = \prod_i q(h_i \mid x) .$$

Lähestymistavan etu on parametrittomuudessa. Tällöin jakaumaa ei tarvitse oppia parametrien kautta, vaan optimoinnin aikana oikea jakauma löydetään ositussääntöjen (*factorization constraints*) kautta. Diskreetetien piilomuuttujien arvoilla edetään perinteisten optimointimenetelmien kautta (eli minkä?), kun taas jatkuvien arvojen kanssa hyödynnetään [variaatiolaskentaa](https://en.wikipedia.org/wiki/Calculus_of_variations). Variaatiolaskenta on myös variaatio-oppimisen nimen pohjana.

Vaihtelevan vapaan energian maksimointi voidaan nähdä myös jakauman $q$ Kullback-Leibler-eroavuuden (*divergence, KL-divergence*) $D_{KL}$ minimointina suhteessa jakaumaan $p$, sillä 

$$\mathcal{L}(x,\theta,q) = \log p(x;\theta)-D_{KL}(q(h \mid x) \mid \mid p(h \mid x;\theta)).$$

Intuitiivisesti KL-eroavuus on etäisyysmitta kahden jakauman välillä, joskaan kyseessä ei ole symmetrinen mitta. Tällöin on siis merkityksellistä se, mitä verrataan ja mihin. KL-eroavuus esitellään tarkemmin kirjan aliluvun 3.13 lopussa. Tässä tapauksessa tavoitteena on $q$:n $p$:n suhteessa tapahtuvan maksimoinnin sijasta (huiput huipuiksi) vastakkainen operaatio, $q$:n $p$:n suhteessa tapahtuva minimointi (laaksot laaksoiksi), sillä näin $q$:n ositussäännöillä voidaan saavuttaa laskentaetua.

#### Discrete Latent Variables

Variaatioinferenssi diskreettien piilomuuttujien arvojen kanssa on suoraviivaista. Tiettyjen $q$:n arvojen esitystavan (taulu tms.) valinnan jälkeen optimoidaan jakauman $q$ parametrit eli piilomuuttujat vaikkapa jyrkimmän gradienttien laskun menetelmällä. Optimoinnin on oltava nopeaa, sillä sitä toistetaan koulutuksessa useasti. Suosittu tapa on ratkaista kunkin piilomuuttujan kohdalla

$$ \frac{\delta}{\delta\hat{h}_i}\mathcal{L}=0.$$

Seuraavaksi esitellään variaatioinferenssin käyttöä binnärisessä harvan koodauksen mallissa. Kirjoittajat kuitenkin toteavat, että syvemmän tason matemaattinen käsittely on kiinnostavaa vain niille, jotka haluavat täysin selvittää menetelmän toiminnan perusteitaan myöten. Halutessaan sen voi siis lukea, mutta konseptin ylätason ymmärtämiseen se ei välttämättä tarjoa mitään uutta.

#### Calculus of Variations

Koneoppimisessa kohdefunktion $J(\theta)$ minimointi tapahtuu etsimällä kriittiset nollakohdat $\Delta_\theta J(\theta)=0$ monimuuttuja-analyysin ja lineaarialgebran avulla. Joissain tapauksissa itse funktion $f(x)$ ratkaisu on kuitenkin kiinnostavampaa, milloin variaatilaskenta tarjoaa sopivat työkalut etenemiseen. Tällöin käytetään funktionaalin eli funktion funktion derivaattoja $ \frac{\delta}{\delta f(x)}J $ variaatioderivaattojen laskentaan. Näitä variaatioderivaattoja hyödynnetään sitten funktion $f(x)$ eli mallin optimoinnissa suhteessa dataan eli näytteisiin $x$ etsimällä jokaisen gradientin nollakohta.

#### Continuous Latent Variables

Variaatioinferenssiä on mahdollista suorittaa myös jatkuvia arvoja saavien piilomuuttujien kanssa. Tällöin on käytettävä variaatiolaskentaa. Yleisratkaisu on käyttää tällöin $q$:na keskiarvokenttäjakaumaa (ks. 19.4) ja normalisoimalla kyseinen jakauma. Variaatiolaskentaa tarvitaan sellaisenaan käytännössä vasta siinä vaiheessa, kun tahdotaan käyttää jotain perinteisestä variaatio-oppimisen menetelmästä poikkeavaa oppimisstrategiaa. Muutoin normalisoitu keskiarvokenttäjakauma tuottaa itsessään kaivattuja odotusarvoja.

#### Interactions between Learning and Inference

Likimääräisen inferenssin käyttö koulutusalgoritmin osana vaikuttaa oppimiseen ja sen kautta itse inferenssialgoritmiin eli malliin. Koulutusalgoritmi pyrkii näet sovittumaan malliin siten, että likimääräiset oletukset esimerkiksi jakaumasta muuttuvat tosiksi. Toisin sanoen koulutus vahvistaa havaittuja ilmiöitä, jolloin suuren todennäköisyyden piiloyksiköt kasvavat todennäköisyydessä ja matalan taas pienenevät. Variaatio-oppimisen koulutuksen lähtökohta sanelee siis myös jokseenkin lopputulosta, ainakin piilomuuttujien jakauman suhteen. Tällöin on tärkeä erottaa toisistaan absoluuttinen tarkkuus ja mallin sovittuvuus - variaatio-oppimisella koulutettu ja sovittunut malli ei vielä kerro, kuinka hyvin sovitus istuu todellisten havaintojen jakaumaan. Missä määrin eroa on sitten taas on, on hankala selvitettävä.

### 19.5 Learned Approximate Inference

Tarkkojen iteratiivisten optimointiprosessien käyttö koulutuksen askelien sisäisinä vaiheina on laskennallisesti raskasta ja aikaa vievää. Siksi monesti optimointia lähestytään likiarvojen kautta, jolloin optimointi nähdään likimääräisesti vaihtelevan vapaan energian maksimoivan jakauman tuottavan funktion oppimisena. Kun koko optimointi nähdään tällä tavoin, voidaan itse funktion opettelua lähestyä neuroverkon kautta.

#### Wake-Sleep

Likimääräisen piilomuuttujien jakauman oppimisen keskeisenä ongelmana on oikeiden vastauksien puute ohjattua oppimista ajatellen. Tätä varten on kehitetty torkkualgoritmi (*wake-sleep*), joka nostaa näytteitä sekä itse näkyvien muuttujien että piilomuuttujien jakaumasta. Varsinaisilla näytteillä koulutettaisiin tällöin piilomuuttujia, kun taas piilomuuttujista näytteistetyillä näytteillä koulutettaisiin näkyviä muuttujia. Näin malli oppii sekä varsinaisten näytteiden ja piilomuuttujien suhteen, että sen, mitkä piilomuuttujat tuottavat varsinaisia näytteitä vastaavia näytteitä.

Koulutetun inferenssiverkon käyttäminen on huomattu olevan iteratiivista inferenssiä tehokkaampaa. Se on muodostunut myös eniten käytetyimmäksi menetelmäksi generatiivisessa mallintamisessa. Likimääräisellä inferenssillä on mahdollista kouluttaa laaja kirjo malleja, joita esitellään seuraavassa ja samalla viimeisessä luvussa.

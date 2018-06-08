
*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#13.-Linear-Factor-Models" data-toc-modified-id="13.-Linear-Factor-Models-1">13. Linear Factor Models</a></span><ul class="toc-item"><li><span><a href="#13.1-Probabilistic-PCA-and-Factor-Analysis" data-toc-modified-id="13.1-Probabilistic-PCA-and-Factor-Analysis-1.1">13.1 Probabilistic PCA and Factor Analysis</a></span></li><li><span><a href="#13.2-Independent-Component-Analysis-(ICA)" data-toc-modified-id="13.2-Independent-Component-Analysis-(ICA)-1.2">13.2 Independent Component Analysis (ICA)</a></span></li><li><span><a href="#13.3-Slow-Feature-Analysis" data-toc-modified-id="13.3-Slow-Feature-Analysis-1.3">13.3 Slow Feature Analysis</a></span></li><li><span><a href="#13.4-Sparse-Coding" data-toc-modified-id="13.4-Sparse-Coding-1.4">13.4 Sparse Coding</a></span></li><li><span><a href="#13.5-Manifold-Interpretation-of-PCA" data-toc-modified-id="13.5-Manifold-Interpretation-of-PCA-1.5">13.5 Manifold Interpretation of PCA</a></span></li></ul></li></ul></div>

## 13. Linear Factor Models

Tässä luvussa käsitellään todennäköisyysmalleja, jotka tunnetaan nimellä lineaariset faktorimallit (*linear factor models*). Näiden mallien toimintaperiaate on etenkin generatiivisten mallien pohjana. Lineaarinen faktorimalli käyttää kohinaa syötteenmuuntofunktiossaan $x \to h$, jossa $h$ tarkoittaa piilomuuttujia (*latent variables*). Nämä mallit auttavat yksinkertaisuudessaan selittävimpien piirteiden löytämisessä.

Yksinkertaisuutensa vuoksi nämä mallit olivatkin ensimmäisiä aihepiirin tutkittujen mallien joukossa. Mallit etsivät ensin datantuottoprosessia eniten kuvaavimmat piilopiirteet $h$. Tämän jälkeen tuotetaan uusia, opittuun datantuottoprosessiin pohjaavia näytteitä yhdessä mallin parametrien sekä kohinan kanssa kaavan 

$$ x = Wh + b + \text{noise} $$ 

mukaisesti. Kaavan $W$ merkitsee mallin parametreja ja $b$ vääristymää.

### 13.1 Probabilistic PCA and Factor Analysis

Todennäköisyyteen pohjaava pääkomponenttianalyysi (*probabilistic principal component analysis, PCA*) on edellä määritellyn kaavan erityistapaus muiden vastaavien faktorianalyysimenetelmien rinnalla. Erot ovat piilomuuttujien kohinan ja alustuksen valinnassa. Piilomuuttujien tehtävänä on tavoittaa syötteen piirteiden riippuvuudet. 

Faktorianalyysissä kohina on vain normaalijakaumasta näytteistettyä ja syötteen piirteiden oletetaan olevan ehdollisesti riippumattomia piilomuuttujien $h$ tapauksessa. Kohinan oletetaan tulevan diagonaalisesti normaalijakauman kovarianssimatriisista, jossa varianssit ovat syötteen muuttujakohtaisia variansseja. Tällöin faktorianalyysi tuottaa näytteitä jakaumasta

$$x \tilde{} \mathcal{N}(x;b, WW^T + \text{diag}(\sigma^2)),$$

jossa $\text{diag}(\sigma^2)$ on kovarianssimatriisin diagonaali.

Matka tästä todennkäisyyteen pohjaavaan pääkomponenttianalyysiin on lyhyt. Varianssien diagonaalin sijasta käytetäänkin yhtä varianssia skalaarina parametrien identtiteettimatriisin kertoimena:

$$x \tilde{} \mathcal{N}(x;b, WW^T + \sigma^2I),$$

mikä voidaan ilmaista myös toisin

$$ x= Wh+b+\sigma z .$$

Jälkimmäisen kaavan $z$ on normaalijakaumasta $\mathcal{N}(z;0,I)$ tuotettua kohinaa. Kun $\sigma \to 0$, lähenee todennäköisyyteen pohjaava PCA normaalin PCA:n toimintatapaa.

PCA:n toiminta perustuu havaintoon siitä, että piilomuuttujilla $h$ kyetään poimimaan useimmat datan variaation lähteet. Poiminta ei ole täydellistä, minkä vuoksi menetelmään liittyy olennaisesti rekonstruointivirhe (*reconstruction error*) $\sigma^2$.

### 13.2 Independent Component Analysis (ICA)

Riippumattomien komponenttien analyysi on yksi vanhimmista piirreoppimisen algoritmeja. Menetelmä pyrkii löytämään täysin riippumattomia piilomuuttujia, joilla datantuottoprosessia voidaan mallintaa. Nimitys on yleinen monille toisistaan eroaville menetelmille. Faktori- ja pääkomponenttianalyysiin nähden läheisin on kuitenkin sellainen, jossa piilomuuttujien jakauma $p(h)$ päätetään etukäteen ja sen jälkeen opitaan deterministisesti $x = Wh$.

Pohja-ajatus liittyy piilomuttujien jakauman valintaan. Kun se on lähtökohtaisesti datasta riippumaton, oletuksena on, että datasta löydetään helpommin myös riippumattomia piirteitä. Tällöin tavoillaan sellaisten datan sisällä olevien piilevien signaalien löytämistä, joista yhdessä voidaan muodostaa mahdollisimman häviötön alkuperäinen syötesignaali.

Tällä menetelmällä voidaan eristää esimerkiksi yksittäisten ihmisten ääniä ihmisjoukon puheensorinasta. Menetelmää käytetään myös aivoaaltojen erotteluun. Sensoreita on kuitenkin oltava useita, joista sitten menetelmä osaa yhdessä eristää yksittäiset äänet. 

 Menetelmästä on, kuten sanottua, useita variantteja eri tarkoituksiin. Kaikille on kuitenkin yhteistä vaatimus jakauman $p(h)$ ei-normaaliudesta (*non-Gaussian*), sillä vain tällöin voidaan taata piilomuuttujien $h$ riippumattomuus. Tässä kohden ero muihin lineaarisiin faktorimenetelmiin on selkein. Tavallista on valita jakauma, joka painottuu lähelle nollaa. Tällöin on samoin tavallista, että menetelmällä tuotetaan harvoja piirteitä.

ICA-menetelmät eivät myöskään tästä syystä ole generatiivisia kirjan tarkoittamassa mielessä. Koska menetelmät eivät kykene kuvantamaan jakaumaa $p(h)$, ne eivät myöskään kykene tällöin kuvantamaan jakaumaa $p(x)$, mistä jälkimmäisin on generatiivisten mallien vaatimus. ICA-menetelmät ovat ennenkaikkea signaalien erottelumenetelmiä.

ICA-menetelmät voidaan kuitenkin yleistää moniksi erilaisiksi epälineaarisiksi generatiivisiksi malleiksi. NICE (*nonlinear independent components estimation*) pyrkii generatiivisuuteen sarjalla enkoodereita ja dekoodereita. Riippumaton osa-avaruuden analyysi (*independent subspace analysis*) taas sallii riippuvuuksia ryhmien sisällä, muttei niiden välillä. Menetelmää on käytetty myös kuvien piirteiden kanssa.

### 13.3 Slow Feature Analysis

Hidas piirreanalyysi (*SFA*) on ajan huomioiva lineaarinen faktorimalli. Perusajatuksena on, että merkittävimmät piirteet vaihtuvat hitaiten. Videoissa pikselikohtaiset arvot muuttuvat nopeasti, mutta kohtauksissa esiintyvät hahmot eivät niinkään. 

Menetelmä perustuu kahden aika-askeleen $t$ ja $t+1$ väliseen virhefunktioon $L(f(x^{(t+1)}), f(x^{(t)}))$ ja sen regularisointiin. Funktio $f$ on piirteitä erotteleva funktio. ICAn tavoin SFA ei ole suoranaisesti generatiivinen malli selkeän jakauman $p(x)$ puuttumisen vuoksi. 

Menetelmä koulutetaan löytämään lineaariset parametrit $\theta$, joilla aika-askelten erotus minimoidaan. Vaatimuksina on, että piirteet ovat nolla-keskiarvoisia ja yksikköhajontaisia ratkaistavuuden vuoksi. Samoin peräkkäisten opittavien piirteiden on oltava lineaarisesti toisistaan riipumattomia. Tällöin menetelmä on ratkaistavissa algebrallisesti. Tällöin sen tulokset on myös ennakoitavissa.

Tyypillistä on, että syötedataa muunnetaan jollain epälineaarisella funktiolla ennen menetelmään syöttämistä. Menetelmiä voidaan myös ketjuttaa siten, edeltävä mallin ulostulo muunnetaan syötteen tavoin ja syötetään seuraavaan SFA-malliin. Näin voidaan oppia biologisissakin järjestelmissä, kuten V1-korteksissa tai tilan hahmotuksen alueissa havaittuja piirteitä. Menetelmä ei kuitenkaan ole yleistynyt tutkimuksessa.

### 13.4 Sparse Coding

Harva koodaus (*sparse coding*) on paljon tutkittu ohjaamaton piirteiden poimintamenetelmä. Menetelmän nimi viittaa etenkin piilomuuttujien $h$ muodostukseen, mutta sitä käytetään myös yleisesti menetelmästä kokonaisuutena. Jälleen muihin lineaarisiin faktorimalleihin ero tulee $p(h)$ (piikki lähellä nollaa, esim. Studentin *t*-jakauma) ja kohinan jakaumien valinnassa (vaimennettu termillä $\frac{1}{\beta}$).

Koulutus tapahtuu enkooderi-dekooderi-tyyppisesti. Enkooderilla opitaan dataa, kun taas dekooderilla opitaan tuottamaan dataa vastaavia näytteitä. Parametrien sijaan menetelmä perustuu optimointiongelman ratkaisuun, jossa etsitään parhaiten sopivia piilomuuttujia $h$. Optimointi tehdään suhteessa valittuun jakaumaan $p(h)$. Menetelmän harvuus on seurausta optimoinnissa käytetystä $L^1$-normista, joskin harvuus näkyy ennemmin poimituissa piirteissä kuin itse $h$:n arvoissa.

Ei-parametrisen enkooderin kanssa yhdistettynä harvan koodauksen menetelmät kykenevät minimoimaan rekonstruointivirheen tehokkaimmin. Koska kyseessä on ei-parametrinen lähestymistapa, ei parametreja tarvitse tällöin opetella. Täten malli ei tuota myöskään erikseen minimoitavaa testivirhettä. Siksi harva koodaus yhdessä ei-parametrisen enkooderin kanssa on hyvä piirteiden poiminnan työkalu osana suurempia parametrisia malleja.

Tutkimuksissa on itseasiassa osoitettu, että harvaa koodausta käyttäen kyetään saavuttamaan parempi yleistysvirhe hahmontunnistuksen tehtävissä. Etenkin tapauksissa, joissa luokkakohtaisia näytteitä on hyvin harvakseltaa, menetelmää hyödyäntäen voidaan silti saavuttaa tietoa yleistyvimmistä piirteistä.

Ei-parametrisuutensa vuoksi menetelmän käyttö on kuitenkin laskennallisesti raskasta. Kyseessä ei ole yrityksen ja erehdyksen kautta etenevä asteittainen koulutus vaan iteratiivinen funktion ratkaisu. Parametriset vastaavat mallit ovat usein verraten matalia, minkä vuoksi niiden koulutus on nopeampaa. Harvan koodauksen käyttö backpropia hyödyntävässä mallissa on myös haasteellista.

Menetelmän tuottamat näytteet (kyseessä on kuitenkin generatiivinen malli) ovat usein huonolaatuisia. Menetelmä oppii yksittäiset piirteet todella hyvin, mutta menetelmän luonteesta johtuen piirteiden yhdistäminen on satunnaisempaa.

### 13.5 Manifold Interpretation of PCA

Lineaariset faktorimallit voidaan nähdä myös piirreavaruuden pinnan oppivina algoritmeina. Todennäköisyyteen pohjaava PCA voidaan esimerkiksi nähdä piirteiden muodostaman tason rajaavana alueena. Mikäli PCA:n löytämien piirteiden muodostama taso on kaksiulotteinen, muodostaa PCA ikäänkuin litistetyn pallon tasolle, josta voidaan tulkita kohina yhdessä kahden piirteen yhteisvaikutukseen. Näin menetelmä rajaa näistä piirteistä datantuottoprosessiin liittyvän todennäköisimmän alueen.


*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#18.-Confronting-the-Partition-Function" data-toc-modified-id="18.-Confronting-the-Partition-Function-1">18. Confronting the Partition Function</a></span><ul class="toc-item"><li><span><a href="#18.1-The-Log-Likelihood-Gradient" data-toc-modified-id="18.1-The-Log-Likelihood-Gradient-1.1">18.1 The Log-Likelihood Gradient</a></span></li><li><span><a href="#18.2-Stochastic-Maximum-Likelihood-and-Contrastive-Divergence" data-toc-modified-id="18.2-Stochastic-Maximum-Likelihood-and-Contrastive-Divergence-1.2">18.2 Stochastic Maximum Likelihood and Contrastive Divergence</a></span></li><li><span><a href="#18.3-Pseudolikelihood" data-toc-modified-id="18.3-Pseudolikelihood-1.3">18.3 Pseudolikelihood</a></span></li><li><span><a href="#18.4-Score-Matching-and-Ratio-Matching" data-toc-modified-id="18.4-Score-Matching-and-Ratio-Matching-1.4">18.4 Score Matching and Ratio Matching</a></span></li><li><span><a href="#18.5-Denoising-Score-Matching" data-toc-modified-id="18.5-Denoising-Score-Matching-1.5">18.5 Denoising Score Matching</a></span></li><li><span><a href="#18.6-Noise-Contrastive-Estimation" data-toc-modified-id="18.6-Noise-Contrastive-Estimation-1.6">18.6 Noise-Contrastive Estimation</a></span></li><li><span><a href="#18.7-Estimating-the-Partition-Function" data-toc-modified-id="18.7-Estimating-the-Partition-Function-1.7">18.7 Estimating the Partition Function</a></span><ul class="toc-item"><li><span><a href="#Annealed-Importance-Sampling" data-toc-modified-id="Annealed-Importance-Sampling-1.7.1">Annealed Importance Sampling</a></span></li><li><span><a href="#Bridge-Sampling" data-toc-modified-id="Bridge-Sampling-1.7.2">Bridge Sampling</a></span></li></ul></li></ul></li></ul></div>

## 18. Confronting the Partition Function

Suuntaamattomat graafi- eli todennäköisyysmallit käyttävät tunnusomaisesti normalisoimatonta todennäköisyysjakaumaa $\tilde{p}(x; \theta)$. Normalisointiin tarvitaan ositusfunktiota $Z(\theta)$, jotta mallilla saadaan tuotettua kelvollinen todennäköisyysjakauma

$$ p(x;\theta) = \frac{\tilde{p}(x; \theta)}{Z(\theta)}. $$

Jotkin mallit kiertävät normalisoinnin tarvetta joko eritystekniikoin tai tekemällä todennäköisyysjakauman $p$ selvittämisen turhaksi. Tässä luvussa keskitytään etenkin niihin malleihin, joiden käyttö edellyttää hankalan todennäköisyysjakauman $p$ muodostamista.

### 18.1 The Log-Likelihood Gradient

Suuntaamattomien mallien koulutus [suurimman uskottavuuden](https://fi.wikipedia.org/wiki/Suurimman_uskottavuuden_estimointi) estimoinnilla on hankalaa, sillä ositusfunktio $Z$ on silloin riippuvainen mallin parametreista. Tällöin gradientit lasketaan hajoittamalla todennäköisyysfunktion gradientit sekä normalisoimattoman todennäköisyysjakauman että ositusfunktion gradienteiksi:

$$ \Delta_\theta \log p(x;\theta) = \Delta_\theta \log \tilde{p}(x;\theta) - \Delta_\theta \log Z(\theta).$$

Nämä kaksi ositusta tai vaihetta tunnetaan myös positiivisena ja negatiivisena oppimisvaiheena (*learning phase*). Negatiivinen oppimisvaihe on suuntaamattomille malleille vaikein, mistä RBM on tyypillinen esimerkki.

Intuition tasolla oppimisvaiheet voidaan nähdä ikäänkuin regularisointina. Positiivisessa vaiheessa näytteen muuttujien todennäköisyyttä kasvatetaan, kun taas negatiivisessa vaiheessa regularisoidaan. Näin malli oppii pikkuhiljaa jakauman $p_{model}$, joka sovittuu jakaumaan $p_{data}$ painottamalla paremmin sovittuvia muuttujia ja vaimentamalla muita.

Ositusfunktion gradienttien kaava johdetaan kirjassa auki kaavoissa 18.5-13, mutta lopullisessa muodossaan se on

$$\mathbb{E}_{x \tilde{} p(x)} \Delta_\theta \log \tilde{p}(x).$$

Toisin sanoen ositusfunktion gradientti on odotusarvo todellisen datantuottojakauman näytteillä lasketusta normalisoimattoman todennäköisyysjakauman gradienteista.

### 18.2 Stochastic Maximum Likelihood and Contrastive Divergence

Ositusfunktion gradientit $\Delta_\theta \log Z$ voidaan laskea naaivisti ajatellen käyttämällä Markovin ketjuja gradienttien laskennan yhteydessä, sillä Markovin ketjuilla saadaan normalisoimattomista jakaumista approksimoitua normalisoituja ja täten koulututksessa käyttökelpoisia jakaumia. Käytännössä tämä lähestymistapa on kuitenkin laskennallisesti erittäin kallis, sillä Markovin ketjut on koulutettava tasapainotilaan aina jokaisen gradienttien laskennan yhteydessä. Lähtökohtana se silti toimii.

Positiivista ja negativiista vaihetta on luonnehdittu myös koneoppimisen ulkopuolella. Yleinen ajatus on, että positiivisessa vaiheessa malli oppii todellisuudesta sellaisena, kuin se ilmenee. Negatiivisessa vaiheessa malli käy läpi vääriä oletuksiaan (*false beliefs*) todellisuudesta ja korjaa niitä. Esimerkkinä tästä käytetään uneksintaa, jossa unien näkeminen nähdään ikäänkuin todellisuuden käsittelynä vääristyneen käsityksen eli jakauman kautta. Unet ja niiden tapahtumat olisivat tällöin jossain määrin vääriä uskomuksia nostettuna opitusta jakaumasta.

Yksi tavoista keventää Markovin ketjujen laskennallista taakkaa on alustaa ne lähelle mallin jakaumaa, jolloin sisäänajo ei kestä kauaa. Vastakohtaisen eroavuuden (*contrastive divergence*) algoritmi alustaa Markovin ketjun varsinaisen datan jakaumasta nostetuilla näytteillä. Tavoitteena on etenkin vaimentaa jakauman vääriä uskomuksia, eli niitä osia, jotka eivät vastaa oikeaa, opittavaa jakaumaa. 

Menetelmä kykenee muokkaamaan jakauman niitä osia, joista se saa tietoa näytteiden kautta, mutta näytteettömät jakauman alueet jäävät vähemmälle huomiolle. Jälkimmäisiä alueita kutsutaan harhaanjohtaviksi moodeiksi (*spurious modes*). Sen on myös havaittu lisäävän mallin vääristymää. Kepeytensä vuoksi sitä suositellaankin joko alustusmenetelmänä, käytettäväksi esikoulutukseen ennen raskaamman ja tarkemman MCMC-mallin käyttöä. 

Syvien mallien koulutukseen siitä ei kuitenkaan perinteisessä toteutusmuodossaan ole juuri suoraa hyötyä, sillä syvien mallien piilokerroksille ei juuri ole mahdollista nostaa "oikean" jakauman näytteitä alustusta varten. Piilokerroksille on jokaiselle ajettava oma Markovin ketjunsa sisään.

Harhaanjohtavien moodien syntymistä voidaan ehkäistä käytämällä stokastinen suurimman uskottavuuden (*stochastic maximum likelihood, SML*) eli sitkeän (*persistent*) vastakohtaisen eroavuuden menetelmällä, jossa opittavaa jakaumaa päivitetään pienin askelin ja useasti käyttäen edellisiä tiloja seuraavien lähtökohtana (vrt. SGD ja optimointi).  SML:n avulla Markovin ketjut kykenevät käymään tehokkaasti läpi laajan määrän moodeja ja tilojen ketjujen ylläpitämisen johdosta myös piilokerrosten käsittely mahdollistuu. SML:ää voidaan siis käyttää syvien mallien kanssa tehokkaasti.

SML voi kuitenkin tulla epätarkaksi, mikäli etemisvauhti on Markovin ketjujen miksaantumista nopeampi. Teoreettista ratkaisua tälle ongelmalle ei ole, vaan etenemisen askelpituus ja Markovin ketjujen sovitusiteraatioiden määrä on kokeiltava tapauskohtaisesti. SML:ää voidaan kouluttaa myös useammalla parametrijoukolla siten, että mallia koulutetaan sekä nopeilla muutoksilla (korkea oppimiskerroin) ja moodeihin keskittyen (matala oppimiskerroin). Lopuksi eri tavoin koulutetut vastaavat parametrit lasketaan esimerkiksi yhteen.

### 18.3 Pseudolikelihood

Suoran ositusfunktion laskennan lisäksi Monte Carlo -approksimointia voidaan lähestyä myös epäsuorasti. Suuntaamattomissa graafimalleissa todennäköisyyksien suhteita on helppo hyödyntää, sillä 

$$ \frac{p(x_1)}{p(x_2)}=\frac{\frac{1}{Z}p(x_1)}{\frac{1}{Z}p(x_2)}=\frac{\tilde{p}(x_1)}{\tilde{p}(x_2)}. $$

Toisin sanoen ositusfunktiota ei tarvitse laskea. Tämä on näennäisuskottavuudeksi (*pseudolikelihood*) kutsutun lähestymistavan pohjana. 

Intuitiivisesti näennäisuskottavuudessa lasketaan ehdollinen todennäköisyys jollekin *piirteelle* $x_i$ suhteessa kaikkiin muihin piirteisiin $x_{-i}$. Kaikkien piirteiden osalta näennäisuskottavuus on

$$ \sum^n_{i=1} \log p(x_i \mid x_{-i}). $$

Laskenta on huomattavasti tehokkaampaa verrattuna ositusfunktioon, sillä laskuja tarvitsee tehdä vain $n$ piirteen ja $k$ piirrekohtaisen mahdollisen arvon kanssa $k \times n$ kertaa. Ositusfunktiota laskiessa sama tehokkuusluku on $k^n$. Näennäisuskottavuus vaatii kuitenkin riittävän tarkasti toimiakseen riittävän suuren näytemäärän.

Mikäli tavoitellaan riittävää tarkkuutta ehdollisten todennäköisyyksien laskentaan, on näennöisuskottavuus kelvollinen algoritmi. Kokonaisen todennäköisyysjakauman $p(x)$ määrittämiseen se ei kuitenkaan ole kovin hyvä. Samoin menetelmän käyttö on kyllä mahdollista syvien mallien kanssa, mutta vain matalien tai matemaattisesti yhteensopivia inferenssimenetelmiä hyödyntävien sellaisten tapauksissa. Laskennallisesti menetelmä on mahdollista tuoda SML:n tasolle oikein säätämällä.

### 18.4 Score Matching and Ratio Matching

Ositusfunktion laskentaa voidaan välttää myös pisteiden parituksilla (*matching*). Pisteytys lasketaan derivaatan logaritmisen todennäköisyyden $\Delta_x \log p(x)$ ja syötteen $x$ suhteen. Tavoitteena on minimoida pisteitä kautta linjan. Menetelmä toimii vain, mikäli normalisoimattoman todennäköisyysjakauman derivaatat on laskettavissa suoraan, sillä menetelmä perustuu osaderivaattojen laskentaan (vrt. optimoinointi). Esimerkiksi harvan koodauksen tai syvien Boltzmannin koneiden kanssa menetelmää ei voida käyttää. Pisteytystä ei voida käyttää, mikäli mallissa on joko näkyvissä tai piilotetuissa kerroksissa diskreettejä arvoja.

Pisteytyksen jatkokehitelmä, joka sallii myös diskreettejä arvoja, on suhteiden paritus. Diskreettien arvojen tulee olla tällöin binäärisiä. Kyseinen menetelmä välttää edelleen ositusfunktion laskentaa ja sen on huomattu toimivan hyvin esimerkiksi kohinan poiston kanssa kuvia käytettäessä. Menetelmä voidaan intuition tasolla ymmärtää samaan toimivaksi näennäisuskottavuuden kanssa. Korkeaulotteisen ja harvan datan kanssa suhteiden paritus toimii hyvin (esim. sanavektorit).

### 18.5 Denoising Score Matching

Joskus todellisen datantuottojakauman $p_{data}$ tavoittelun sijasta kiinnostavampaa on tasoitetun jakauman $p_{smoothed}$ tuottaminen. Tämä tehdään lisäämällä kohinaa koulutuksessa käytettyyn dataan, sillä vain käytössä olevilla näytteillä ei välttämättä kyetä tavoittamaan todellista datantuottojakaumaa. Menetelmä vastaa vahvasti autoenkooderien toimintaa, ja tähän viitataankin autoenkoodereja käsittelevässä luvussa.

### 18.6 Noise-Contrastive Estimation

Ositusfunktiota estimoivat mallit eivät usein laske itse ositusfunktiota suoraan, vaan joko sen gradienttia (SML, vastakohtainen eroavuus) tai eivät ollenkaan (näennäisuskottavuus). Kohinaan vertaava estimointi (*noise-contrastive estimation*) eroaa näistä laskemalla ositusfunktiolle suoran likimäärän. Tällöin kaavaksi tulee

$$ \log p_{model}(x) = \log \tilde{p}_{model}(x;\theta) + c,$$

jossa $c \approx -\log Z(\theta)$. Erittelyn molempia osia päivitetään koulutuksen yhteydessä. Koulutuksen aikana mallille syötetään joko kohinaa tai varsinaisia syötteitä. Koulutus on ohjaamattoman koulutuksen sijasta ohjattua ja malli opetetaankin erottamaan oikeat näytteet kohinasta.

Menetelmä toimii parhaiten, kun muuttujien määrä on rajoitettu. Muuttujilla voi olla laajakin kirjo eri arvoja, mutta muuttujien määrän kasvaessa menetelmä alkaa kärsiä. Samoin kohinajakauma on kaksiteräinen miekka. Helposti käsiteltävä kohinajakauma voi olla liian ilmeisesti itse datasta erottuva, jolloin malli tyytyy liian karkeisiin tuloksiin ja erottelee kohinaa tehokkaasti opittuaan datasta vasta karkeat alkeet. Menetelmän toimivuuden rajoitteet ovat samat, kuin pisteiden parituksen ja näennäisuskottavuuden menetelmillä.

Mikäli menetelmää käytetään siten, että kohinan jakaumana käytetään mallin jakaumaa ja jakauma päivitetään ennen gradienttien päivittämistä, ollaan tekemisissä itseensä vertaavan estimoinnin (*self-contrastive estimation*) kanssa. Sen gradientin odotusarvo vastaa suurimman uskottavuuden avulla saatavaa gradientin odotusarvoa.

Menetelmän perusajatuksena on, että hyvän generatiivisen mallin on kyettävä erottamaan kohina varsinaisesta datasta. Edelliseen läheisesti liittyvä ajatus on, että hyvä generatiivinen malli kykenee tuottamaan näytteitä, joita mikään luokittelija ei kykene erottamaan oikeista näytteistä. Nämä kaksi ajatusta ovat kirjan lopussa esiteltävän GAN-verkon pohjana.

### 18.7 Estimating the Partition Function

Tässä aliluvussa käsitellään menetelmiä, joilla saadaan suoraan likimääräistettyä ositusfunktio, mikä on tarpeellista normalisoidun datan jakauman laskemisessa suuntaamattomien graafimallien kanssa, etenkin itse mallien arvioinnissa. Kahden samankaltaisen maksimiuskottavuutta käyttävän mallin vertaaminen keskenään edellyttää näet ositusfunktion olemassaoloa. Laskennallisista haasteista johtuen yksinkertaisten Monte Carlo -menetelmien käyttö ei auta, sillä käytännössä data on sekä liian moniulotteista että useamman moodin omaavaa.

#### Annealed Importance Sampling

Mikäli kahden toisiinsa verrattavan mallin jakaumat ovat kaukana toisistaan, voidaan välijakaumilla (*intermediary distributions*) rakentaa siltaa näiden välille. Tätä kutsutaan karkaistuksi merkityksellisyyden mukaan näytteistyksesti (*annealed importance sampling*). Tällöin ositusfunktiota lähdetään määrittämään jostain tunnetusta lähtökohdasta (esim. nolla-painoinen RBM) ja edetään muutos kerrallaan kohti tavoitejakaumaa.

Välijakaumat, kuten myös lähtöjakauma, ovat vapaasti valittavissa. Yleisratkaisu on käyttää painotettua geometristä keskiarvoa lähde- ja kohdejakaumien välillä. Tämän jälkeen määritellään Markovin ketju siten, että siirtymäfunktiolla $T$ päästään välijakaumasta toiseen. Kun Markovin ketju on sisäänajettu ja välijakaumat oikein painotettu, on lähde- ja kohdejakaumien ositusfunktioiden suhde mahdollista laskea. Tällä suhteella saadaan selvitettyä kohdejakauman ositusfunktion suuruusluokka, mitä voidaan sitten käyttää normalisoidun jakauman laskentaan.

#### Bridge Sampling

Karkaistua merkityksellisyyden mukaan näytteistystä vastaava menetelmä on siltanäytteistys (*bridge sampling*). Useiden välijakaumien sijasta lähde- ja kohdejakaumien välillä käytetään vain yhtä siltajakaumaa. Tavoitteena on edelleen tunnetun ositusfunktion omaavan lähdejakauman ja ositusfunktion osalta selvityksen kohteena olevan kohdejakauman ositusfunktioiden suhteen selvittäminen.

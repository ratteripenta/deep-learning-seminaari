
*Created by Petteri Nevavuori.*

---

# Deep Learning seminaari

Kirjana Goodfellow et al.: Deep Learning (2016)

Otsikot seuraavat pääotsikoiden tasolla kirjaa, mutta alaotsikot eivät aina.

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#12.-Applications" data-toc-modified-id="12.-Applications-1">12. Applications</a></span><ul class="toc-item"><li><span><a href="#12.1-Large-Scale-Deep-Learning" data-toc-modified-id="12.1-Large-Scale-Deep-Learning-1.1">12.1 Large-Scale Deep Learning</a></span><ul class="toc-item"><li><span><a href="#Fast-CPU-Implementations" data-toc-modified-id="Fast-CPU-Implementations-1.1.1">Fast CPU Implementations</a></span></li><li><span><a href="#GPU-Implementations" data-toc-modified-id="GPU-Implementations-1.1.2">GPU Implementations</a></span></li><li><span><a href="#Large-Scale-Distributed-Implementations" data-toc-modified-id="Large-Scale-Distributed-Implementations-1.1.3">Large-Scale Distributed Implementations</a></span></li><li><span><a href="#Model-Compression" data-toc-modified-id="Model-Compression-1.1.4">Model Compression</a></span></li><li><span><a href="#Dynamic-Structure" data-toc-modified-id="Dynamic-Structure-1.1.5">Dynamic Structure</a></span></li><li><span><a href="#Specialized-Hardware-Implementations-of-Deep-Networks" data-toc-modified-id="Specialized-Hardware-Implementations-of-Deep-Networks-1.1.6">Specialized Hardware Implementations of Deep Networks</a></span></li></ul></li><li><span><a href="#12.2-Computer-Vision" data-toc-modified-id="12.2-Computer-Vision-1.2">12.2 Computer Vision</a></span><ul class="toc-item"><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-1.2.1">Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Contrast-Normalization" data-toc-modified-id="Contrast-Normalization-1.2.1.1">Contrast Normalization</a></span></li><li><span><a href="#Dataset-Augmentation" data-toc-modified-id="Dataset-Augmentation-1.2.1.2">Dataset Augmentation</a></span></li></ul></li></ul></li><li><span><a href="#12.3-Speech-Recognition" data-toc-modified-id="12.3-Speech-Recognition-1.3">12.3 Speech Recognition</a></span></li><li><span><a href="#12.4-Natural-Language-Processing" data-toc-modified-id="12.4-Natural-Language-Processing-1.4">12.4 Natural Language Processing</a></span><ul class="toc-item"><li><span><a href="#$n$-grams" data-toc-modified-id="$n$-grams-1.4.1">$n$-grams</a></span></li><li><span><a href="#Neural-Language-Models" data-toc-modified-id="Neural-Language-Models-1.4.2">Neural Language Models</a></span></li><li><span><a href="#High-Dimensional-Outputs" data-toc-modified-id="High-Dimensional-Outputs-1.4.3">High-Dimensional Outputs</a></span><ul class="toc-item"><li><span><a href="#Use-of-a-Short-List" data-toc-modified-id="Use-of-a-Short-List-1.4.3.1">Use of a Short List</a></span></li><li><span><a href="#Hierarchical-Softmax" data-toc-modified-id="Hierarchical-Softmax-1.4.3.2">Hierarchical Softmax</a></span></li><li><span><a href="#Importance-Sampling" data-toc-modified-id="Importance-Sampling-1.4.3.3">Importance Sampling</a></span></li><li><span><a href="#Noise-Contrastive-Estimation-and-Ranking-Loss" data-toc-modified-id="Noise-Contrastive-Estimation-and-Ranking-Loss-1.4.3.4">Noise-Contrastive Estimation and Ranking Loss</a></span></li></ul></li><li><span><a href="#Combining-Neural-Language-Models-with-$n$-grams" data-toc-modified-id="Combining-Neural-Language-Models-with-$n$-grams-1.4.4">Combining Neural Language Models with $n$-grams</a></span></li><li><span><a href="#Neural-Machine-Translation" data-toc-modified-id="Neural-Machine-Translation-1.4.5">Neural Machine Translation</a></span><ul class="toc-item"><li><span><a href="#Using-an-Attention-Mechanism-and-Aligning-Pieces-of-Data" data-toc-modified-id="Using-an-Attention-Mechanism-and-Aligning-Pieces-of-Data-1.4.5.1">Using an Attention Mechanism and Aligning Pieces of Data</a></span></li></ul></li><li><span><a href="#Historical-Perspective" data-toc-modified-id="Historical-Perspective-1.4.6">Historical Perspective</a></span></li></ul></li><li><span><a href="#12.5-Other-Applications" data-toc-modified-id="12.5-Other-Applications-1.5">12.5 Other Applications</a></span><ul class="toc-item"><li><span><a href="#Recommender-Systems" data-toc-modified-id="Recommender-Systems-1.5.1">Recommender Systems</a></span><ul class="toc-item"><li><span><a href="#Exploration-versus-Exploitation" data-toc-modified-id="Exploration-versus-Exploitation-1.5.1.1">Exploration versus Exploitation</a></span></li></ul></li><li><span><a href="#Knowledge-Representation,-Reasoning-and-Question-Answering" data-toc-modified-id="Knowledge-Representation,-Reasoning-and-Question-Answering-1.5.2">Knowledge Representation, Reasoning and Question Answering</a></span><ul class="toc-item"><li><span><a href="#Knwoledge,-Relations-and-Question-Answering" data-toc-modified-id="Knwoledge,-Relations-and-Question-Answering-1.5.2.1">Knwoledge, Relations and Question Answering</a></span></li></ul></li></ul></li></ul></li></ul></div>

## 12. Applications

Tässä luvussa käydään läpi syväoppivien menetelmien käytännön sovelluksia. Ensin käydään läpi ison mittakaavan (*large-scale*) neuroverkkoimplementaatioita. Sen jälkeen käydään läpi muutama yksittäinen käytännön sovellus.

### 12.1 Large-Scale Deep Learning

Eräs merkittävin neuroverkkojen suorituskykyyn vaikuttanut yksittäinen asia on niiden koko ja sen kasvu aina 1980-luvulta alkaen. Koon kasvaessa kasvavat myös vaatimukset laskentaa suorittavan raudan sekä laskentakomponenttien infrastrutktuurin osalta.

#### Fast CPU Implementations

Neuroverkkojen koulutus alkoi käyttämällä yhden koneen prosessoria, CPU:ta. Nykyisin koulutukseen käytetään joko suoraan näytöonhjaimia (*graphical processing unit, GPU*) tai sitten hajautetun laskennan keinoin monen koneen prosessoria samanaikaisesti. Tämä ei kuitenkaan tarkoita, etteikö vain prosessorilla koulutettavaksi suunniteltuja neuroverkkoja olisi mahdollista tehdä. Tällöin on mentävä koodin alimmille tasoille ja käytettävä tiettyjä laskentaoperaatiota tarkkaan rajattujen prosessorien kanssa.

#### GPU Implementations

Useimmat nykyaikaiset neuroverkot koulutetaan kuitenkin näytönohjaimilla. Samat asiat, jotka peleissä vaativat suorituskykyä, auttavat myös neuroverkkojen koulutuksessa. Pelien renderöinti vaatii monien yksinkertaisten laskutoimitusten rinnakkaista laskentakykyä. Tehokkaampi näytönohjain takaa näiden laskutoimitusten nopeamman suorittamisen. Sama näytönohjainten ominaisuus on suoraan hyödynnettävissä myös monikerroksisten verkkojen koulutuksessa. Siksi näytönohjainten käyttö on yleistynyt myös syväoppimisen alueella.

Laskentatehon lisäksi näytönohjaimilla on prosessoriin nähden käytössään nopeampaa muistia, jonka käyttö laskutoimitusten puskurina auttaa tehostamaan verkon koulutusta. Näytönohjainten käyttö syväoppivien verkkojen koulutuksessa sai alkunsa 2000-luvun puolivälissä ja etenkin Nvidian CUDA-kielen tulon jälkeen 2010-luku on ollut näytönohjainpohjaisen verkkojen koulutuksen aikaa.

Näytönohjainten muistin toimintalogiikka eroaa prosessoreista, mutta ei ole tämän kirjan aiheena. Siitä syystä GPU-koodaamiseen ei ole suotavaa lähteä, vaan käyttää esimerkiksi CUDA-yhteensopivia ylemmän abstraktiotason kirjastoja näytönohjaimella suoritettavaan koulutukseen. Näitä kirjastoja ovat Pythonilla mm. TensorFlow (Google), Theano ja PyTorch (Facebook).

#### Large-Scale Distributed Implementations

Usein yhden tietokoneen resurssit eivät riitä koulutuksen tarpeisiin. Tällöin koulutuksen taakkaa tulisi jakaa (*distribute*) useiden koneiden välillä. Ensinnäkin dataa voidaan jakaa eri koneille, jolloin kyseessä on datan rinnakkaisuus (*data parallelism*). Samoin yksi malli voidaan ositella ja jakaa osa-alueittain useammalla koneelle (*model parallelism*).

Mallin jakaminen on kahdesta mainitusta helpompaa. Sitä voidaan suoraan käyttää sekä koulutukseen että ulostulojen laskentaan. Datan jakaminen on hankalampaa etenkin koulutuksen osalta, sillä silloin yhden ja saman mallin päivittäminen monilta eri koneilta samanakaisesti ilman ristiriitoja vaatii erityistoimenpiteitä etenkin gradienttien laskennan suhteen.

Eräs ratkaisu tähän on asynkroninen SGD (*asynchronous stochastic gradient descent*). Tällöin muutama laskentayksikköä vastuutetaan mallin parametrien päivittämisestä, jolloin parametreille muodostetaan käytännössä oma palvelimensa (*server*). Muut laskentayksiköt sekä lukevat että kirjoittavat tietoja tähän palvelimeen. Kaupalliset toteutukset perustuvat usein tähän ratkaisuun, kun taas akateemiset resurssivajeensa vuoksi keskittyvät muihin (GPU).

#### Model Compression

Kaupallisissa ratkaisuissa usein mallin käyttämisen nopeus on sen kouluttamisen nopeutta tärkeämpää. Näin on esimerkiksi tapauksissa, joissa samaa mallia käytetään jopa miljoonienkin käyttäjien laitteilla (mobiili). Tällöin käyttäjien laitteet asettavat rajoja laskennallisille resursseille. Tällöin käytetään mallin tiivistämistä (*model compression*).

Koska jokainen mallin parametreista ei ole merkityksellisimpien piirteiden mallintamisen kannalta tärkeää, voidaan mallia karsia. Tällöin koulutus tehdään kaksivaiheisesti. Ensin koulutetaan isompi malli perinteiseen tapaan mahdollisimman tarkaksi. Tämän jälkeen koulutetun mallin ulostuloista rakennetaan uusi koulutussetti, jolla koulutetaan sitten pienempi malli.

#### Dynamic Structure

Koska neuroverkkojen yksiköt ovat yhteydessä vain osiin syötteistä, on kaikkien yksiköiden laskeminen jokaisella syötteellä tehotonta. Siksi laskenta vain kulloinkin syötteen kannalta merkityksellisimpien yksiköiden kanssa tehostaa mallin toimintaa huomattavasti. Tätä kutsutaan dynaamiseksi rakenteeksi (*dynamic structure*) ja joskus ehdolliseksi laskennaksi (*conditional computation*).

Syväoppivien menetelmien tapauksessa tämä on toteutettavissa mallisarjoilla (*cascade of models*). Esimerkiksi harvinaisen tapauksen luokittelu on hyödyllistä aloittaa karkeasti joko toteamalla selkeästi että sitä ei havaittu tai että se voi havainnosta löytyä. Tämä on ensimmäinen karkea malli. Mikäli tämä antaa tulokseksi ei, ei laskentaa jatketa ja tulos on selvä. Ehkä-tuloksen kanssa laskentaa jatketaan tarkemmalla ja tarkemmalla mallilla aina niin pitkään, kunnes varmuus on riittävä. 

Karkeat mallit ovat matalan kapasiteetin nopeasti ajettavia malleja. Sarjassa edettäessä mallien kapasiteetti ja täten laskennallinen kompleksisuus kasvaa. Kun mallit ovat lähtökohtaisesti poissulkevia, on laskenta kuitenkin harvinaisissa tapauksissa tehokasta nopean seulonnan vuoksi. Googlen StreetView käyttää kahden mallin sarjaa ensin numeroiden havaitsemiseen ja sitten tunnistamiseen.

Perinteisen koneoppimisen päätöspuiden (*decision tree*) myös neuroverkkoja voidaan suunnitella dynaamisesti tarkentuviksi. Tällöin käytetään yhtä hallinnoivaa porttiverkkoa (*gater network*), joka ohjaa sitten laskennan jollekin erikoistuneemmalle ja kompleksimmalle mallille (*expert network*). Vain yhden erikoistuneen verkon valinta helpottaa laskentaa, kun syötettä ei tarvitse laskea jokaisessa verkossa erikseen. Tämä toimii kuitenkin parhaiten vain porttiverkon vaihtoehtojen määrän ollessa pieni.

Dynaamisen rakenteen käyttö on kuitenkin hankalaa yhdessä hajautetun laskennan kanssa. Tällöin etenkin rinnakkaislaskenta on vaikeaa, sillä monimutkaisemman portitetun koulutuksen riittävän tehokas toteutus hajautetusti on osoittautunut vaikeaksi.

#### Specialized Hardware Implementations of Deep Networks

Algoritmisten ratkaisujen lisäksi on koulutuksen tehokkuutta tavoiteltu myös erikoistuneiden laitteiden suunnittelulla. Merkittävin tätä kehitystä ajanut tekijä on itsenäisesti toimivien syväoppivien menetelmien teollisten ja käytännön sovellusten lisääntyminen. Erikoistuneiden piirien (ASIC, FPGA) lisäksi etenkin prosessorien ja näytönohjainten rinnankytkentöihin perustuvat järjestelmät ovat tuoneet viimeaikojen merkittävämpiä suorituskykylisäyksiä. Toisaalta myös lukujen tarkkuuden vähentäminen nopeuttaa laskentaa, ja tämä on ollut myös 2010-luvun tutkimusten aiheena.

### 12.2 Computer Vision

Näkemällä havaitseminen on ihmisille ja eläimille helppo mutta koneille haastava tehtävä. Konenäkö onkin siksi syväoppivien menetelmien tutkimuksen aiheista yksi perinteisimmistä. Kasvojen ja tekstin tunnistamisen lisäksi myös mm. äänien havaitseminen visuaalisesti värähtelevistä esineistä ovat konenäön alueen sovelluksia. Suurin työ on tehty kuitenkin ihmisen kykyjen mallintamisessa erinäisten asioiden prosessoinnin automatisoimiseksi.

#### Preprocessing

Muista menetelmistä poiketen kuvantunnistuksen datan esikäsittely on melko kevyttä. Tärkeintä on, että kuvien arvot on skaalattu johonkin helposti hyödynnettävään väliin *kautta linjan*. Tällainen väli on esimerkiksi $[0,1]$. Samoin kuvien tulee lähtökohtaisesti olla samankokoisia, joten kuvat pitää rajata tai skaalata samaan kokoon, joskaan tämä ei aina ole pakollista.

Varsinaisen datan prosessoinnin sijasta datasetin kasvattaminen (*augmentation*) on konenäön tehtävissä usein merkityksellisempää, joskin tämä koskee vain koulutusdatasettiä. Näin saadaan etenkin yleistysvirhettä laskettua merkittävästi useimpien konenäkömallien tapauksessa. Muutakin esiprosessointia voidaan datalle tehdä, mutta riittävän suuren datasetin kanssa tämä on usein tarpeetonta. Tällöin riittää usein vain normalisointi tai skaalaus.

##### Contrast Normalization

Kuvissa esiintyvä kontrasti, eli tummimpien ja vaaleimpien pikselien erotuksen suuruus, on merkityksetön varianssin lähde ja siksi turvallisesti poistettavissa. Kontrastin poistolle esitellään kaksi tapaa, koko kuvaan (*global*) ja paikallisiin (*local*) arvoihin pohjaava kontrastin normalisointi. Kontrastilla tarkoitetaan koneoppimisen yhteydessä pikselikohtaisten arvojen hajontaa.

Koko kuvan normalisointi tehdään vähentämällä kuvan pikseleiden arvosta kuvakohtainen keskiarvo ja sen jälkeen skaalaamalla data siten, että koko kuvan pikseleiden keskihajonta pysyy samana. Skaalaus tehdään suhteen kuvan $L^2$ normiin käyttäen mahdollisesti nollalla jakamista estäviä pienikokoisia lisäparametreja.

Tämä ei kuitenkaan ole menetelmänä ongelmaton, sillä näin toimittaessa vahvistetaan ennemmin kuvan kohinaa tai kuva-artefakteja. Samoin halutut piirteet, kuten reunat, eivät vahvistu tällä menetelmällä.

Paikallinen normalisointi tehdään vain kuvaan ositellen. Paikallista normalisointia voidaan tehdä monin keinoin, mutta päätavoite on kuvantunnistuksen kannalta merkityksellisten asioiden, eli reunojen, korostaminen. Koko kuvan normalisoinnin sijasta paikallinen normalisointi voidaan suorittaa osana konvoluutiota kuvantunnistumallia koulutettaessa.

##### Dataset Augmentation

Kuvantunnistusluokittimen yleistymistä on helppo parantaa kasvattamalla koulutusdatasetin kokoa kasvattamalla sitä erilaisin geometrisin muunnoksin olemassaolevista kuvista. Näin opitut piirteet pakotetaan robustimmiksi, jolloin malli tavoittaa paremmin opituista luokista yleistettävimmät piirteet.

### 12.3 Speech Recognition

Puheentunnistuksessa koneoppimistehtävänä on akustisen signaalin eli ääniaallon muuntaminen tekstimuotoisiksi sanoiksi ja lauseiksi. Oppiminen voidaan tehdä joko ennalta määriteltyjen piirteiden avulla tai suoraan raakadatasta oppien. Vaikka neuroverkkojen suorituskyky oli 1990-luvun taitteessa muihin menetelmiin pohjaavien mallien tasolla, vasta syvien verkkojen käyttäminen 2010-luvun kynnyksellä toi neuroverkot kunnolla puheentunnistuksen piiriin.

Käytetyt mallit olivat rajoitettuja Boltzmannin koneita, jotka esitellään tarkemmin kirjan kolmannessa osassa. Joka tapauksessa ohjatut mallit rakennettiin käyttämällä ohjaamattomia malleja ohjatun mallin kerrosten koulutukseen. Syvien verkkojen käyttö paransi pitkään muihin menetelmiin pohjanneiden mallien tarkkuuksia yleisesti käytetyllä TIMIT-datasetillä. 

Ymmärryksen ja menetelmien kehityksen myötä ohjaamattomasta esikoulutuksesta kuitenkin luovuttiin ja malleissa siirryttiin ReLU-yksiköitä ja poispudotusta hyödyntäviin konvoluutioverkkoihin. Nämä verkot käsittelivät äänidataa spektrogrammikuvana, jossa yksi suunta kuvaa aikaa ja toinen taajuuksien voimakkuuksia. Tutkimus etenee tällä hetkellä LSTM-verkkojen hyödyntämisen alueella.

### 12.4 Natural Language Processing

Luonnollisen kielen prosessointi (*natural language processing, NLP*) keskitty koneiden helposti käytettävien kielten sijasta ihmisten käyttämien kielien käsittelyyn. Käytännön sovellusalueita ovat mm. konekääntäminen kieleltä toiselle ja tunteiden tunnistus tekstistä. Monet NLP-sovellukset perustuvat kielen, sanojen ja kirjainten todennäköisyysjakaumiin.

Kielen kanssa toimittaessa on keskityttävä sekvenssejä käsitteleviin malleihin, jolloin lauseita käsitellään sanojen sekvensseinä kirjainten sijasta. Sanojen paljouden vuoksi sanapohjaiset datasetit ovat todella harvoja, ja tätä on pyritty jossain määrin tehostamaan.

#### $n$-grams

Kielimalli (*language model*) määrittelee todennäköisyysjakauman sekvenssin diskreeteille symboleille (*token*), joita voivat olla sanat, merkit ja bitit. Ensimmäisiä toimivia kielimalleja kutsuttiin $n$-grammeiksi niiden symbolimäärän $n$ perusteella. Nämä mallit määrittävät symboleiden ehdollisen todennäköisyyden edellisten symboleiden suhteen.

Näiden mallien vaatiman suurimman todennäköisyysarvion (*maximum likelihood estimate*) laskeminen on suoraviivaista selvittämällä kunkin $n$-grammin esiintymismäärä datasetissä. Erityistapaukset on nimetty seuraavasti:

- *unigram*, kun $n=1$.
- *bigram*, kun $n=2$.
- *trigram*, kun $n=3$.

Näitä malleja on samanaikaisesti käytössä enemmän kuin yksi. Usein koulutetaankin mallit $n$ ja $n-1$g grammeille samanaikaisesti seuraavan kaavan mukaisesti:

$$ P(x_t \mid x_{t-n+1}, \text{ ... } ,x_{t-1})=
\frac{P_n(x_t \mid x_{t-n+1}, \text{ ... } ,x_{t})}{P_{n-1}(x_t \mid x_{t-n+1}, \text{ ... } ,x_{t-1})}.$$

Kaavan $P$ on yhden symbolin ehdollinen todennäköisyys, $P_n$ on $n$-symbolisen sekvenssin ($n$-grammi) todennäisyys ja $P_{n-1}$ taas $n-1$ pituisen sekvenssin ($n-1$-grammi) todennäköisyys.

Perustavanlaatuisena ongelmana näissä malleissa on kuitenkin niiden todennäköisyyksien pienuus. Koska $P_n$ lasketaan koko datasetille, on sen todennäköisyys etenkin sekvenssin pituutta kasvatettaessa koko ajan pienempi. Samoin *unigrammeille* $P_{n-1}$ on määrittämätön. Keinoja tämän välttämiseksi on luonnollisesti kehitetty useampia. Toinen perustavanlaatuinen ongelma on alttius piirreavaruuden kasvun hallitsemattomille seurauksille (*curse of dimensionality*).

Näitä sekvenssin esiintymisen todennäköisiä malleja on pyritty kehittämään esimerkiksi luokkapohjaisilla kielimalleilla (*class-based language models*). Nämä mallit ryhmittelevät ensin sanoja, minkä jälkeen todennäköisyyksiä lasketaan näitä ryhmiä käyttäen. Näin kuitenkin menetetään paljon yksittäisten sanojen mukanaan kantamasta informaatiosta.

#### Neural Language Models

Neuraaliset kielimallit (*neural language models, NLM*) pyrkivät ratkaisemaan $n$-grammimallien piirreavaruuden suuruuteen liittyvät ongelmat. Ne kykenevät luokkapohjaisiin malleihin verrattuna tunnistamaan kahden sanan samankaltaisuuden ilman, että ne hävittävät sanojen eroja. NLM-mallit kykenevät myös säilyttämään saman kontekstin jakavien sanojen tilastollisen linkin sisäisen hajautetun jakauman ansiosta, jonka ansiosta samoja piirteitä sisältäviä sanoja kohdellaan samaan tapaan.

Esimerkkinä käytetään sanoja `koira` ja `kissa`, jotka malli on oppinut liittämään samankaltaisiin konteksteihin. Näin jomman kumman sanan esiintyessä malli saa samalla myös tietoa toisen sanan mahdollisesta esiintymisestä. Jaetut piirteet määrittyvät etenkin sen suhteen, missä määrin jotkin sanat esiintyvät samassa yhteydessä eli kontekstissa. Näin vältetään ulottovuuksien kirousta, sillä oppiminen ei ole enää sana- vaan kontekstikohtaista.

Kontekstipohjaisia sanojen esitystapoja kutsutaan sanojen upotukseksi (*word embedding*). Tällöin kukin symboli nähdään aluksi sanastoavaruuden (*vocabulary space*) yhtenä pisteenä, missä yksi sana vastaa yhtä ulottuvuutta. Koulutuksen edetessä malli oppii upottamaan yksittäisiä sanoja alempipiirteiseen samankaltaisia sanoja ryhmittelevään avaruuteen.

Samankaltaista upottamista tapahtuu myös esimerkiksi konvoluutioverkkojen ja kuvien tapauksessa, mutta luonnollisen kielen yhteydessä ajatus on kiinnostavampi. Tämä johtuu etenkin siitä, että kuvat on ilmaistavissa pohjimmiltaan numeroin, kun taas sanat ja niiden merkitykset eivät niinkään. Tärkeässä roolissa on joka tapauksessa neuroverkkojen piilokerrokset ja siellä opittavat yhteydet.

#### High-Dimensional Outputs

Sanoja tuottavien mallien koulutuksessa ja käytössä ongelmaksi muodostuvat todella suuret ja harvat sanastomatriisit, joissa voi olla jopa satoja tuhansia sanoja. Näiden matriisien kertominen esimerkiksi ulostuloa laskennassa on äärimmäisen epäkäytännöllistä sekä laskenta- että muistiresurssien osalta ja muodostuu yksittäiseksi suurimmaksi pullonkaulaksi.

##### Use of a Short List

Ensimmäiset luonnollisen kielen mallit päättivät kiertää liian suuren sanaston ongelmaa rajoittamalla sanastoa vain kymmeniin tuhansiin. Mikäli $\mathbb{V}$ on sanasto, rakennettiin lyhyt lista (*short list*) $\mathbb{L}$ valitsemalla vain useimmin esiintyvät sanat. Tämän rinnalle otettiin myös loput eli harvinaisemmat sanat omaan häntäjoukkoonsa $\mathbb{T}$. Tällöin neuroverkon tehtävänä on sanan ennustamisen lisäksi ennustaa sen kuuluminen häntäjoukkoon.

Tämän menetelmän ongelmana on kuitenkin sen yleistymiskyky vain yleisimpien sanojen joukkoon valittuihin sanoihin. Sanamäärällisesti suurin osa jää tällöin mallin yleistymiskyvyn ulkopuolelle. Tämä myös heikentää näin toimivan mallin käytettävyyttä huomattavasti.

##### Hierarchical Softmax

Toinen lähestymistapa suuren $\mathbb{V}$:n käsittely hierarkisesti. Tällöin yksittäisten sanojen sijasta käsitelläänkin sanojen, kategorioiden kategorioita puumaiseen tapaan, jolloin laskennallisia vaatimuksia saadaan tiputettua $O({\mid\mathbb{V}\mid}) \to O(\log{\mid\mathbb{V}\mid})$ täysin tasapainotetun binääripuun tapauksessa. Sanakohtaisen kontekstirikkauden lisäämiseksi yhteen sanaan voidaan rakentaa myös useampi polku. Tällöin yhden sanan todennäköisyys on kaikkien siihen johtavien polkujen yhteistodennäköisyys.

Ehdollisten todennäköisyyksien ennustamiseksi puun jokaisessa solmussa käytetään logistista regressiota käyttäen samaa kontekstia $C$ kautta linjan. Koulutuksen lähtökohta on ohjattu oppiminen, jolloin malli kykenee oppimaan esimerkkien avulla. Hierarkisen menetelmän tehokkuuden vuoksi myös itse mallin optimointi tehokkaampaa koko mallin osalta.

Vaikka itse sanoihin $\mathbb{V}$ johtava puu olisikin mahdollista optimoida kunkin sanan esiintymistodennäköisyyden avulla, ei siihen käytetty vaiva tuo lopulta kovin merkittäviä laskennallisia säästöjä. Puun optimointi tuo laskennallisesti vain lineaarisia säästöjä sen tehokkuuden ollessa $O(n)$. Itse verkon koulutus on laskennallisesti kuitenkin luokkaa $O(n^2)$, jolloin se vaikuttaa koulutukseen huomattavasti enemmän.

Tämän menetelmän avoimena kysymyksenä on itse hierarkiapuun määrittäminen. Olemassaolevien hierarkioiden lisäksi ne voidaan myös oppia mallin kouluttamisen yhteydessä. Helppoa se ei silti ole sen diskreetin ja siksi gradienttipohjaisen ulottumattomissa olevan luonteen vuoksi. Samoin tämän menetelmän käyttö ylipäätään tuottaa heikohkoja testituloksia.

##### Importance Sampling

Sanoja voidaan myös jättää huomiotta gradientteja laskiessa ja näin pienentää laskettavien gradienttien määrää. Tämä tapahtuu jättämällä selkeästi väärät sanat pois, konteksti huomioiden. Sanojen alustava merkintä todennäköisiksi ja epätodennäköisiksi voi kuitenkin olla laskennallisesti verraten raskaskin operaatio, minkä vuoksi on parempi käyttää pienempiä sanaston osajoukkoja.

Lähtökohtana käytetään *uni*- tai *bigrammilla* muodostettuja sanakohtaisia merkitsevyyden (*importance*) lähtökohtajakaumia. Näitä jakaumia hyödyntäen voidaan kullekin sanalle laskea sen jälkeen tehokkaasti gradientit kielimallin koulutuksen edistyessä. Malli koulutetaan nostamalla näytteitä näitä jakaumia hyödyntäen käyttäen sekä positiviisia että negatiivisia näytteitä.

Menetelmä toimii ylipätään tilanteissa, joissa ollaan tekemisissä harvojen vektorien tai matriisien kanssa. Sanojen säkki (*bag of words*) on yksi tällainen vektoriesimerkki, jossa merkitään sanan esiintymistä dokumentissa. Merkityksellisten näytteiden (*importance sampling*) menetelmä esitellään tarkemmin myöhemmin.

##### Noise-Contrastive Estimation and Ranking Loss

Todennäköisyyksien sijaan sanoja voidaan myös pisteyttää. Näin toimittaessa malli opetetaan pisteyttämään sanat kulloinenkin syöte huomoiden. Mitä suurempi pistemäärä, sitä todennäköisempää sen sanan ennustaminen on. Näin itse ehdollisen todennäköisyyden laskeminen on kuitenkin vaikeaa. Kirjan myöhemmässä vaiheessa esitellään tämän ajatuksen kehityksen tulos, kohinaan vertaamiseen (*noise-contrastive*) perustuva toimivaksi osoittautunut menetelmä.

#### Combining Neural Language Models with $n$-grams

Neuroverkkoihin verrattuna $n$-grammeihin pohjaavat mallit saavuttavat tehokkaasti todella suuren kapasiteetit tallettamalla monien sanalistojen esiintymistodennäköisyyksiä. Jos näiden sanalistojen hakuun käytetään vaikkapa puurakenteita, on niiden käyttö vieläkin tehokkaampaa ja riippumattomampaa kapasiteetista. Neuroverkkojen kohdalla on toisin, sillä kapasiteetin lisäys lisää perinteisesti laskenta-aikaa.

Käyttämällä näitä kahta toisistaan eroaavaa mallia ensemble-tyyppisesti voidaan kuitenkin saavuttaa parempia testituloksia. Ensemble-mallien osamallit toimivat itsenäisesti ja pohja-ajatuksena on, että monta mallia on yhdessä parempi keskimääräinen arvaus kuin yksi.

Yhdistämistapoja on useita. Rinnakkaisia malleja voi ensinnäkin olla yksinkertaisesti hyvin monia. Samoin $n$-grammimalleja voidaan käyttää myös syötteenä neuroverkolle. Joka tapauksessa näiden keinojen tavoitteena on lisätä neuroverkkomallin kapasiteettia samalla kasvattaen laskennallisia vaatimuksia vain minimaalisesti.

#### Neural Machine Translation

Neuroverkoilla tehtävä konekääntäminen (*neural machine translation*) on ihmisten käyttämän kahden eri kielen eri sanoilla ilmaistujen samojen merkityksien oppimista. Järjestelminä nämä mallit ovat monikomponenttisia. Usein ensin on karkeampi eri käännösvaihtoehtoja ehdottava malli, minkä jälkeen ehdotuksia arvioidaan ja suodatetaan seuraavilla malleilla.

Ensimmäisissä konekäännöksen malleissa päädyttiin jo enkooderi-dekooderi-malleihin. Pelkkien sanojen esiintymistodennäköisyyksien ennustamisen sijasta sanojen sekvensseistä sanojen sekvensseihin tapahtuvassa ennustamisessa on enemmän hyötyä ehdollisista todennäköisyyksistä - mikä on kohdekielen lauseen ennusteen $\hat{B}$ todennäköisyys, kun pohjana on lähtökielen lause $A$?

Ensimmäinen $n$-grammien suorituskyvyn ohittanut neuroverkko oli perinteinen myötäkytketty verkko, jonka vaatimuksena oli kiinteän pituiset syötelauseet. Joustavuutta tähän rajoitteeseen saatiin toistavilla neuroverkoilla. Käyttämällä eri verkkoja syötteen lukemiseen ja ennusteen tuottamiseen päästiin paremmin käsiksi kahden eri kielen nyansseihin.

##### Using an Attention Mechanism and Aligning Pieces of Data

Mitä pidemmäksi käännettävät lauseet muodostuvat, sitä vaikempaa niitä on oppia kääntämään suoraviivaisesti. Vaikka tämä olisi toistavilla verkoilla mahdollista pitkän koulutuksen jälkeen, on parempi käyttää huomiomekanismeihin (*attention mechanism*). Tällöin koko syötteen lukemisen jälkeen käännös toteutetaan sana kerrallaan kiinnittäen kunkin sanan kohdalla huomiota sekvenssin eri osiin sanan kääntämiseen vaikuttavan kontekstin huomioimiseksi tehokkaammin.

Huomiomekanismissa on kolme pääkomponenttia:

- **Raakadatan lukija**, joka muuntaa mallille tuotavan syötteen piirteiksi.
- **Muisti**, josta raakadatasta opittuja piirteitä voidaan mielivaltaisesti noutaa.
- **Ennustaja**, joka käyttää hyväkseen muistia ennusteen tuottamiseen sekvenssin alkio kerrallaan.

#### Historical Perspective

Kielellisten symbolien piirteiden oppiminen on esitetty ajatuksena jo 1980-luvun puolivälissä. Riippuen kulloinkin vallalla olevasta tavasta, symbolit tuotiin näihin alkuvaiheen verkkoihin ensin yksittäisinä kirjaimina. Vasta 2000-luvulla siirryttiin kokonaisten sanojen käyttöön, joskin nykytutkimus edistää kumpaakin tapaa. Menetelmien kehittyessä myös käyttökohteet ovat laajentuneet mm. tekstien osien sisällön luokitteluun.

### 12.5 Other Applications

Lopuksi esitellään vielä perinteisistä kuvan-, tekstin- ja puheentunnistuksen alueiden toteutuksista poikkeavia neuroverkkojen hyödyntämisalueita.

#### Recommender Systems

Kyky tehdä suosituksia monien asioiden joukosta on myös koneoppimisen alueella paljon tutkittu alue. Käytännön sovelluksina tällaisia ovat mainonta ja suositukset esimerkiksi kauppojen tai vaikkapa IMDB:n sivuilla. Kyseessä lähes aina ennustaminen tilanteessa, jossa jokin tuote tai mainos on toistaiseksi ennustuksen kohteelle tuntematon mutta hänestä on jotain ennakkotietoa selvillä.

Usein tätä ongelmaa ratkotaan ohjattuna ongelmana siten, että määritellään kiinnostumista ilmaisevat toiminnot ja koulutetaan malleja näitä tietoja käyttäen. Tällöin kyseessä on regressio-ongelman mallintaminen. Tämä ei kuitenkaan ole ainoa lähestymistapa.

Toinen, esimerkiksi elokuvien suosittelussa käytetty menetelmä on yhteistyöhön perustuva suodatus (*collaborative filtering*). Jos tavoitteena on ennustaa kohdekäyttäjän mielipidettä elokuvasta $A$, aloitetaan ensin etsimällä ne käyttäjät, jotka ovat kohdekäyttäjän kanssa samankaltaisia. 

Samat nähdyt ja samoin arvostellut elokuvat ovat yksi tapa löytää nämä. Näistä samankaltaisista käyttäjistä otetaan edelleen vain ne, jotka ovat nähneet kohde-elokuvan, jota kohdekäyttäjällä koitetaan ennustaa. Näistä kohde-elokuvan nähneistä kohdekäyttäjää vastaavista käyttäjistä muodostetaan sen jälkeen ennuste kohdekäyttäjälle.

Tämän tiedon mallintamiseen on monia keinoja. Tärkein esiinnostettu menetelmä on kahden lineaarisen funktion eli bilineaarinen menetelmä, jossa opitaan elokuvien tapauksessa sekä käyttäjien piirteet että elokuvien piirteet erikseen. Tämän jälkeen ennuste muodostetaan käyttämällä vain käyttäjään tai elokuvaan liittyvää kerrointa yhdessä opittujen käyttäjä- ja elokuvapiirteiden kanssa.

Netflixin 2000-luvun puolivälissä järjestämä kilpailu siivitti etenkin suosittelujärjestelmien tutkimusta. Ensimmäiset hyvin suorituneista neuroverkoilla toteutetuista suosittelijaverkoista olivat rajoitettuun Boltzmannin koneeseen pohjaavia. Netflixin kilpailun voittaneen mallin yksi komponenteistakin oli juuri tällainen RBM.

Yhteistyöhön perustuvalla suosituksella on kuitenkin yksi perustavanlaatuinen ongelma. Uusien käyttäjien tai tuotteiden kohdalla puuttuu tyystin niihin liittyvä historiatieto, jonka avulla ne voitaisiin asemoida käyttäen olemassaolevien käyttäjien tai tuotteiden tietoja. Tällöin voidaan käyttää lisämääritteitä, joita historiatiedottomalla järjestelmään lisätyllä olennolla on jo alusta alkaen. Tätä kutsutaan sisältöpohjaiseksi suosittelijajärjestelmäksi (*content-based recommender system*).

##### Exploration versus Exploitation

Suosittelijajärjestelmien opettamisessa on eräs sisäänrakennettu ongelma, joka on läheistäsukua vahvistusoppimisen (*reinforcement learning*) ongelmien kanssa. Ytimessä on ajatus siitä, että järjestelmä on tietyssä mielessä induktioloopissa - vain niitä asioita kerätään suosittelutietoa, joita kerätyllä suosittelutiedolla suositellaan käyttäjille.

Tekoälyyn vahvemmin liitetyn vahvistusoppimisen alueella tämä tunnetaan tuntemattomien alueiden tutkimisen (*exploration*) ja opitun tiedon hyödyntämisen (*exploitation*) vaihtokauppana. Mikäli malli toimii vain parhaimman oppimansa käytännön (*policy*) perusteella, se ei voi oppia mahdollisesti vieläkin parempia käytänteitä. Ongelma on läheistä sukua lokaalien minimien ongelmalle, joskin vahvistuoppimisessa on kyse myös koulutusdatasetin laajentamisesta.

Suosittelijärjestelmien kohdalla ongelma on samankaltainen. Perinteisesti eräs ratkaisu on käyttää satunnaisuutta suosituksien tekemisessä, jolloin suosituksien kanssa saadaan tietoa myös opitun harvinaisemmista tuotteista. Joka tapauksessa näiden järjestelmien optimointi muistuttaa vahvistusoppimismallien optimointia.

#### Knowledge Representation, Reasoning and Question Answering

Viimeisenä osa-alueena esitellään tutkimuksen kohteena aktiivinen sanojen, sanaupotteiden ja faktojen välisten yhteyksien mallintaminen. Tämä on tärkeä alue esimerkiksi hakukoneiden kohdalla.

##### Knwoledge, Relations and Question Answering

Sekä matematiikan että tietojärjestelmien mallintamisen alueelta tuttu entiteettien suhteiden ja ominaisuuksien määrittely liittyy olennaisesti neuroverkoilla tehtävään suhteiden ja ominaisuuksien mallintamiseen. Näiden keskinäisten suhteiden oppiminen ja mallintaminen on tehtävä joko suoraan datasta tai käyttämällä valmiita suhteita valmiiksi mallintavia tietokantoja.

Syötedatasta muodostetaan joko $(entiteetti_1, suhde, entiteetti_2)$ tai $(entiteetti,ominaisuus)$ monikkoja. Näitä käyttämällä koulutetaan mallia pyrkien oppimaan syötedatan osien yhteistodennäköisyys. Käytettävät mallit pohjaavat usein NLP-malleihin. Tällöin sanojen sijasta sijasta muodostetaan entiteettien sekvenssejä, joiden suhteiden ja ominaisuuksien piirteitä yritetään sitten oppia.

Käytännön sovelluksena näitä malleja voidaan käyttää ennustamaan tuntemattomia entiteettien välisiä yhteyksiä. Näiden mallien suorituskyvyn mittaus on kuitenkin hankalaa, sillä tavallisesti datasetit koostuvat vain oikeista yhteyksistä eli positiivisista esimerkeistä. Mikäli malli löytää jonkun poikkeavan yhteyden, on hankala sano suoraan onko kyseessä uuden yhteyden löytö vai selkeästi väärä tulos.
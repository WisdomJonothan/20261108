# Harmful-class swaps, direct arm

## 1. bitcoin / 120273_40 slot 8 delta -0.107

QUERY: What is the logic (if there is) behind linking the nSequence and nLockTime?
<p>The <code>nSequence</code> transaction field was originally intended for a replacement logic, but since it was never well thought out, it was never implemented. However, it has been implemented since the beginning that if the <code>nSequence</code> of all inputs is <code>0xffffffff</code>, then the <code>nLocktime</code> and the absolute time are not activated. Later, some additional features are added like the relative time lock consensus rule by not setting the 32nd bit and the RBF policy rule by setting the value

INCUMBENT (bitcoin/76ef704a_15147.txt, judged relevant):
Relative locktime
In mid-2016, the BIP 0068/112/113 soft fork gave consensus-enforced meaning to some values in the nSequence field[7] that is a part of every transaction input, creating a "relative locktime".[citation needed] This allowed an input to specify the earliest time it can be added to a block based on how long ago the output being spent by that input was included in a block on the block chain.

CHALLENGER (bitcoin/7c9e6c23_144877.txt, unjudged):
The original meaning of nSequence was never properly implemented and the value of nSequence is customarily set to 0xFFFFFFFF in transactions that do not utilize timelocks. For transactions with nLocktime or CHECKLOCKTIMEVERIFY, the nSequence value must be set to less than 231 for the timelock guards to have an effect, as explained below.

VERDICT: 

## 2. bitcoin / 124603_8 slot 8 delta -0.148

QUERY: Question about bitcoin mining difficulty adjustment formula in book mastering bitcoin 3rd edtion
<p>In book &quot;Mastering bitcoin&quot; 3rd, page 278, There is a formula about bitcoin mining difficulty adjustment formula as below:</p>
<pre><code>        New Target = Old Target * (20,160 minutes / Actual Time of Last 2015 Blocks)
</code></pre>
<p>I think it should be:</p>
<pre><code>        New Target = Old Target * (Actual Time of Last 2015 Blocks / 20,160 minutes)
</code></pre>
<p>Anyone has comments about it? Thanks.</p>


INCUMBENT (bitcoin/6c9cebd4_1912.txt, judged relevant):
Target Block Time
Difficulty is tweaked to ensure blocks are mined every 10 minutes, regardless of miner activity.

Adjustment Formula
The difficulty is recalculated using a formula that compares the actual mining time to the expected time.

Network Security
Consistent difficulty ensures Bitcoin’s security by preventing too-fast or too-slow block production. Blockchain development services can assist in understanding and optimizing these processes.

CHALLENGER (bitcoin/cfad6f43_143777.txt, unjudged):
1 . What is Difficulty Adjustment in Bitcoin? 2 . Importance of Mining Difficulty Adjustment in Bitcoin 3 . How Bitcoin Maintains Mining Difficulty? 4 . Regular Adjustments 5 . Hash Rate Monitoring 6 . Target Block Time 7 . Adjustment Formula 8 . Network Security 9 . Factors Influence Bitcoin's Difficulty Adjustment 10 . What Happens If Mining Difficulty is Not Adjusted Properly? 11 . Are there Limits to Difficulty Adjustments? 12 . Adjustment Limits 13 . Minimum Adjustment Interval 14 . Target Block Time 15 . Network Stability 16 . Can Difficulty Changes Affect Bitcoin's Value? 17 . How Does Nadcab Labs Handle Mining Difficulty Challenges?

VERDICT: 

## 3. economics / 47636_213 slot 8 delta -0.193

QUERY: How to explain the meaning of the coefficient of interest of Difference-in-Differences?
<p>Considering a Difference-in-Differences (DiD) equation with a dummy variable:</p>
<p><span class="math-container">$$
y_i = \alpha + \beta D_i + Xit + \varepsilon_i.
$$</span>
While <span class="math-container">$$Xit$$</span>
is a set of covariates.
Let's say the DiD here has the frequency is year and unit level is country. The outcome variable is number of rich people per million (yearly data).</p>
<p>Can I explain that <span class="math-container">$\beta$</span> as below:
Let's say  <span class="math-co

INCUMBENT (economics/e17b159b_10472.txt, judged relevant):
DID is a quasi-experimental design that makes use of longitudinal data from treatment and control groups to obtain an appropriate counterfactual to estimate a causal effect. DID is typically used to estimate the effect of a specific intervention or treatment (such as a passage of law, enactment of policy, or large-scale program implementation) by comparing the changes in outcomes over time between a population that is enrolled in a program (the intervention group) and a population that is not (the control group).


Figure 1. Difference-in-Difference estimation, graphical explanation

DID is used in observational settings where exchangeability cannot be assumed between the treatment and control groups. DID relies on a less strict exchangeability assumption, i.e., in absence of treatment, the unobserved differences between treatment and control groups arethe same overtime. Hence, Difference-in-difference is a useful technique to use when randomization on the individual level is not possible. DID requires data from pre-/post-intervention, such as cohort or panel data (individual level data over time) or repeated cross-sectional data (individual or group level). The approach removes biases in post-intervention period comparisons between the treatment and control group that could be the result from permanent differences between those groups, as well as biases from comparisons over time in the treatment group that could be the result of trends due to other causes of the outcome.

Causal Effects (Ya=1 – Ya=0)
DID usually is used to estimate the treatment effect on the treated (cau

CHALLENGER (economics/f0470f56_11104.txt, unjudged):
The regression formula is as follows Liaukonytė, Tuchman, and Zhu (2023): \[ y_{ut} = \beta \text{Post}_t + \gamma_u + \gamma_w(t) + \gamma_l + \gamma_g(u)p(t) + \epsilon_{ut} \] \(y_{ut}\): Outcome of interest for unit u in time t. \(\text{Post}_t\): Dummy variable representing a specific post-event period. \(\beta\): Coefficient measuring the average change in the outcome after the event relative to the pre-period. \(\gamma_u\): Fixed effects for each unit. \(\gamma_w(t)\): Time-specific fixed effects to account for periodic variations. \(\gamma_l\): Dummy variable for a specific significant period (e.g., a major event change). \(\gamma_g(u)p(t)\): Group x period fixed effects for flexible trends that may vary across different categories (e.g., geographical regions) and periods. \(\epsilon_{ut}\): Error term. This model can be used to analyze the impact of an event on the outcome of interest while controlling for various fixed effects and time-specific variations, but using units themselves pre-treatment as controls.

VERDICT: 

## 4. history / 55786_792 slot 10 delta -0.113

QUERY: Could a German insult Hitler without being arrested?
<p>According to <a href="https://en.wikipedia.org/wiki/Paul_von_Lettow-Vorbeck#Post-war_career" rel="noreferrer">Wikipedia</a>, in its article about <a href="https://en.wikipedia.org/wiki/Paul_von_Lettow-Vorbeck#Post-war_career" rel="noreferrer">general Paul von Lettow-Vorbeck</a>,</p>

<blockquote>
  <p>Later, when Hitler offered him the ambassadorship to the <a href="https://en.wikipedia.org/wiki/Court_of_St_James%27s" rel="noreferrer">Court of St James's</a> in 1935, he &ldquo;declined with frigid hauteur&rdquo;; the suggestion for the no

INCUMBENT (history/f393c254_110914.txt, judged relevant):
With the outbreak of the First World War, the colonial territories of the warring empires were just as much a part of the conflict as their European counterparts, though far from the trenches and mud. Hopelessly outnumbered and out-gunned, the Imperial German Schutztruppe (Protection Force) of East Africa had a single ace up their sleeve: the leadership of Paul von Lettow-Vorbeck, a man soon to be known as the Lion of Africa, who waged perhaps the most effective guerrilla campaign of the early 20th century and who held the respect of friend, foe, and even the colonial soldiers under his command. Returning from the war a hero, he would be remembered as one of the most well-respected commanders of the era, so popular and influential that even during the reign of the Nazis, he was able to insult Hitler to his face, refuse any positions offered by the regime, and survive their reign of terror with his popularity and honor intact.



CHALLENGER (history/5639e94f_353392.txt, unjudged):
By 1935, the Nazi Party had consolidated its grip on the Third Reich. The Enabling Act and November 1933 s election made Hitler the supreme power in Germany. The Night of the Long Knives saw the party bear its murderous teeth to opposition but the regime s brutality had been established from the outset; Dachau was founded immediately following Hitler s appointment as Chancellor. So it was a very dangerous time for a decorated German war hero to tell Adolf Hitler to go and fuck himself.

VERDICT: 

## 5. history / 56861_715 slot 8 delta -0.123

QUERY: How was the first carpenter&#39;s square made?
<p>A "square" is a tool, typically L or triangle shaped, that gives a craftsperson a quick and accurate 90° angle to use as a baseline in many important trades.  </p>

<p>It's easy to find a right angle if you already have a square, but how was the first square made?  How can someone create a right angle tool "from scratch"?</p>


INCUMBENT (history/847ee096_101352.txt, judged relevant):
In early nineteenth-century references to squares, the eighteen-inch by twelve-inch size is usual, but the tool has been around for centuries. It was used in Europe long before it was first patented in this country in 1819 by one Silas Hawes, a blacksmith in South Shaftsbury, Vermont.



CHALLENGER (history/ddd6f1af_101394.txt, unjudged):
If you can create a straight edge, you can construct a square. You don't need to know any trigonometry beyond the idea that two right-angles make a straight line.

VERDICT: 

## 6. history / 59294_622 slot 9 delta -0.141

QUERY: How many people did the Islamic Caliphate send to the Maghreb during the Arabization of North Africa?
<p>I know that during the Muslim conquest of the North Africa it was more a military colonization than an Arabization an spread of their culture as the Berbers in North Africa kept their culture and were not merged to the Arabs. After that and with the Hilalian invasions to North Africa that were sent by the order of the Fatimid Caliphate, they Arabized North Africa and spread the Arabic culture. My question now is how many military troops and how many nomads were sent to the <a href="https://

INCUMBENT (history/1bf0e210_90058.txt, judged relevant):
During the rule of the Aghlabid dynasty (founded in 800), Arab migration to increased in numbers due to the anti-Kharijite wars. The number of Arab migrants of Ifriqiya concentrated in the army and the cities, mainly Kairouan, has been estimated at 100,000. Most of the Arab migrants came from Syria and Iraq, which from the start supplied numerous migrants to the Maghreb.[13] The organization of the Aghlabid army was largely based on the Arab tribes who settled in Ifriqiya in the late seventh and eighth centuries.[14] These troops were called the jund, descendants of Arab tribesmen who participated in the Muslim conquest of the Maghreb.[15] In 789, descendant of Ali ibn Abi Talib, Idris ibn Abdallah, fled from the Hejaz and arrived in Tangier after the failed revolt against the Abbasids in the Battle of Fakhkh. He founded the Idrisid dynasty, which established control over modern-day Morocco and western Algeria. The Idrisid dynasty played an important role in the early Islamization of the area, and contributed to an increase in Arab migration and Arabization in major urban centers of the western Maghreb.[16] Several Shia Arabs rapidly flocked to Fez, Arabizing the region. Fez experienced large waves of Arab migration, including one which involved 800 Arabs from Al-Andalus in 818 and one which involved 2,000 Arab families from Ifriqiya in 824.[17]



CHALLENGER (history/e687a84a_90182.txt, unjudged):
The 11th century witnessed the most significant wave of Arab migration, surpassing all previous movements. This event unfolded when the Zirid dynasty of Ifriqiya proclaimed its independence from the Fatimid Caliphate of Egypt. In retribution against the Zirids, the Fatimids dispatched large Bedouin Arab tribes, mainly the Banu Hilal and Banu Sulaym, to defeat the Zirids and settle in the Maghreb. These tribes followed a nomadic lifestyle and were originally from the Hejaz and Najd. To encourage the Banu Hilal and Banu Sulaym to migrate to the Maghreb, the Fatimid caliph provided each tribesman with a camel and financial support and assisted them in crossing from the eastern to the westwen bank of the Nile River. The harsh drought and subsequent economic crisis in Egypt at the time further motivated these tribes to relocate to the Maghreb, which had a better economic situation at the time. The Fatimid caliph instructed them to rule the Maghreb instead of the Zirid emir Al-Mu'izz and told them "I have given you the Maghrib and the rule of al-Mu'izz ibn Balkīn as-Sanhājī the runaway slave. You will want for nothing." and told Al-Mu'izz "I have sent you horses and put brave men on them so that God might accomplish a matter already enacted". Upon their arrival in Cyrenaica, the Arab nomads discovered that the region was almost empty of its inhabitants, with only a few Zenata Berbers remaining, most of whom had been largely destroyed by Al-Mu'izz. Estimates suggest that up to 200,000 Hilalian families moved westward out of Egypt. The Banu Sulaym settled in Cyrenaica while the Hil

VERDICT: 

## 7. history / 62736_601 slot 10 delta -0.098

QUERY: How much did Didius Julianus pay to become emperor of Rome?
<p><a href="https://en.wikipedia.org/wiki/Didius_Julianus" rel="noreferrer">According to Wikipedia</a>, Didius Julianus was proclaimed Roman Emperor in AD 193 by the Praetorian Guard after offering them 25,000 sesterces per soldier (outbidding Sulpicianus's offer of 20,000 sesterces each).  But &quot;25,000 sesterces&quot; is just a number.  How much money would that have been, compared to the yearly pay of a typical soldier (or a typical Praetorian)?</p>


INCUMBENT (history/42af6dfc_86682.txt, judged relevant):
#### Severan dynasty
Commodus fell victim to a conspiracy aided by his Praetorian prefect Quintus Aemilius Laetus in 192. The new emperor Pertinax , who took part in the conspiracy, paid the Praetorians a premium of 3,000 denarii; however he was assassinated three months later, on 28 March 193, by a group of Guards due to his refusal to further increase the premium which had already been paid. The Praetorians then put the empire up to auction and Didius Julianus bought the title of emperor. However, the armies of the Danube chose instead the governor of Pannonia Superior , Septimius Severus , who besieged Rome and tricked the Praetorians when they came out unarmed. The Praetorian Guard was dissolved and replaced by men transferred from Septimius's army.
The new Guard of Septimius Severus made their mark against his rival Clodius Albinus at the Battle of Lyon in 197, and accompanied the emperor to the Orient from 197 to 202, then to Britannia from 208 until his death at York in 211.
Caracalla , son of Septimius Severus, lost favour with his troops by assassinating his own brother and co-emperor, Geta, immediately after his succession. Finally, in 217, while on campaign in the Orient , he was assassinated at the instigation of his prefect Macrinus .

CHALLENGER (history/55743778_113887.txt, unjudged):
Military metrics highlight manpower erosion: by the late Empire, Italian recruitment dwindled, forcing reliance on provincial and barbarian foederati , with legionary numbers peaking at  420,000 around 300 CE but quality declining amid desertions and indiscipline. 75  Historical records document  recruitment shortages, as evidenced by edicts compelling coloni to serve and exemptions failing to fill quotas, correlating with demographic contraction. 76  Institutional corruption amplified these vulnerabilities; the  Praetorian Guard , originally an elite bodyguard, auctioned the throne in 193 CE to Didius Julianus for 25,000 sesterces per soldier, exemplifying how proximity to power fostered regicides and bids over merit.

VERDICT: 

## 8. history / 63331_580 slot 10 delta -0.113

QUERY: Did a unit of 40 men hold off Rommel&#39;s &quot;ghost division&quot; for 18 days?
<p>I found the following baffling claim on a Wikipedia <a href="https://en.wikipedia.org/wiki/Chasseurs_Ardennais" rel="noreferrer">article</a> regarding the &quot;Chasseurs Ardennais&quot; in World War II:</p>
<blockquote>
<p>A unit of only 40 men held off the combined arms of the German forces, including General Rommel's &quot;Ghost Division&quot; for 18 days and only surrendered after running out of ammunition, when asked where the others were, they simply replied 'We are all'.<sup>[citation needed]</sup></p>

INCUMBENT (history/dce2ed0b_81577.txt, judged relevant):
It was the crisp morning of 10th May 1940, the very day the Wehrmacht chose to launch their ambitious invasion of Belgium, the Netherlands, and France. In the deep and tangled Ardennes forest near the sleepy village of Bodange, the famed Chasseurs Ardennais were bracing themselves. These rugged light infantry, experts at woodland warfare, had been entrusted with the task of holding this crucial bottleneck on the road from Bastogne to Neufchâteau — a job requiring guile, grit, and a healthy dose of stubbornness.

The Cast of Characters: Units and Command

The 1st Battalion of the Chasseurs Ardennais, under the steady hand of Major Joseph Romeo, was dispersed along a narrow ridge overlooking the main road at Bodange. Romeo was a man of quiet determination, who valued cunning ambushes and the clever use of terrain over headlong charges. His battalion consisted of roughly 600 men — a lean, mean fighting force armed with FN Model 1930 rifles, Bren-like light machine guns, a sprinkling of 47mm anti-tank guns, and an admirable collection of grenades.


Dawn Patrols and The First Skirmishes

At 05:45, as the first pale light sifted through the trees, Lieutenant Pierre Dupont led a forward reconnaissance patrol along the eastern forest edge. Their orders were clear: identify any sign of German troop movement, no matter how subtle.

By 06:10, Dupont’s patrol had spotted the initial vanguard of German scouts — motorcyclists and bicycle troops from the 5th Panzer Division’s reconnaissance battalion — tentatively probing the road. Dupont’s men fell back carefully, avoiding premature eng

CHALLENGER (history/f262b809_315616.txt, unjudged):
Hotton Hysteria: How the Ardennais Made the Germans Sweat (12th May 1940) By the time the 12th of May 1940 dawned over the mist-shrouded Ardennes, the situation was growing increasingly fraught. The German juggernaut had pushed deep into Belgium, but at Hotton   a small town perched on the banks of the River Ourthe   the Belgian defenders, including the ever-resourceful Chasseurs Ardennais, were preparing a resolute stand. This was no mere skirmish; it was a vital clash to block the German advance and prevent the complete envelopment of Allied forces further east.The Cast and TerrainThe defending force comprised the 1st Battalion of the Chasseurs Ardennais, reinforced by elements of the Belgian 2nd Infantry Division. Their commanding officer, Major Joseph Romeo, had once again orchestrated the deployment, favouring the steep ridges overlooking the River Ourthe and the surrounding wooded hills   terrain perfect for defence, ambush, and generally making life difficult for anyone trying to cross a river under fire.Opposing them was the German 4th Panzer Division, with formidable armour and mechanised infantry poised to exploit any weakness. The Germans aimed to seize Hotton and secure the crossings over the Ourthe to facilitate a rapid push toward Bastogne and beyond. Early Morning   Preparing the DefencesAt 05:30, the Chasseurs were in position, their foxholes and trenches dug deep into the hillside. Anti-tank guns those ever-important 47mm models were placed strategically on the approaches to the town, especially the road bridges over the Ourthe. Machine gun nests peppered t

VERDICT: 

## 9. history / 64282_527 slot 9 delta -0.102

QUERY: Did a submarine ever torpedo and sink a destroyer?
<p>The naval <a href="http://dubm.de/en/the-bitter-fight/" rel="nofollow noreferrer">battle</a> of <a href="https://en.wikipedia.org/wiki/German_submarine_U-405" rel="nofollow noreferrer">U-405</a> versus <a href="https://en.wikipedia.org/wiki/USS_Borie_(DD-215)" rel="nofollow noreferrer">USS Borie</a> sank the Borie, but that was due to its own action of ramming U-405, which had a stronger hull to withstand depth pressures.</p>
<p>Did a submarine ever sink a destroyer with a torpedo?  Or with a deck gun?</p>


INCUMBENT (history/d4425bfe_72558.txt, judged relevant):
The French submarine Surcouf was launched in 1929 with two 203 mm/50 Modèle 1924 guns in a turret forward of the conning tower.[1] These were the second largest guns carried by any submarine after the British HMS M1 during the Second World War.[2] The London Naval Treaty of 1930 restricted submarine guns to a maximum of 155 mm (6.1 inches).

In the early part of World War II, German submarine commanders favored the deck gun for similar reasons as their World War I counterparts; the limited number of torpedoes that could be carried, the unreliability of torpedoes, and because their boats could only travel submerged at slow speed for short distances.[3] The deck gun became less effective as convoys became larger and better equipped, and merchant ships were armed. Surfacing also became dangerous in the vicinity of a convoy because of improvements in radar and direction finding. (See Defensively equipped merchant ships (DEMS) and United States Navy Armed Guard). German U-boat deck guns were eventually removed on the order of the supreme commander of the U-boat Arm (BdU) during World War II, and those deck guns that remained were no longer manned. For a few months in 1943, some U-boats operating in the Bay of Biscay were equipped with enhanced anti-aircraft guns (at the trade-off of reduced torpedo loadouts), being known as "U-Flak" boats to be deployed as service escorts for regular U-boats. After the Royal Air Force modified their anti-submarine tactics which made it too dangerous for a submarine to stay on the surface to fight, the U-Flaks were converted back to standard U-bo

CHALLENGER (history/6bb5d348_296742.txt, unjudged):
At dawn on 31 October 1941, she was torpedoed near Iceland   3   by German submarine U-552 commanded by Kapitänleutnant Erich Topp . Reuben James had positioned herself between an ammunition ship in the convoy and the known position of a German   wolfpack U-boats_in_the_Second_World_War) ,  a group of submarines poised to attack the convoy. The destroyer was not flying the ensign of the United States and was in the process of dropping depth charges on another U-boat when she was engaged.   4   Reuben James was hit forward by a torpedo meant for a merchant ship and her entire bow) was blown off when a magazine Naval_magazine) exploded. The bow sank immediately. The aft section floated for five minutes before going down. Of a crew of seven officers and 136 enlisted men, plus one enlisted passenger, 100 were killed. Only 44 enlisted men and no officers survived.   1     3

VERDICT: 

## 10. history / 68426_409 slot 10 delta -0.068

QUERY: When did the Royal Navy start to fill a given rank of admiral with more than one officer?
<p>Admirals in the Royal Navy used to follow the <a href="https://en.wikipedia.org/wiki/Coloured_squadrons_of_the_Royal_Navy#Flag_officers_and_commodores_promotion_pathway_within_squadrons" rel="noreferrer">coloured-squadron system</a>. If I have understood correctly, when this practice first arose in the seventeenth century, there really were precisely nine admirals in the entire navy (excluding the Lord High Admiral). However, by the time of Nelson, there were many more: In 1803 there were 45 admirals, 

INCUMBENT (history/c196cd25_54836.txt, judged relevant):
### Table B: Flag-ranks in 1812
An example is given of an officer holding a rank
in each of the three squadrons; the date refers to the date of his
promotion to that rank (the rank of Admiral of the Red was established
in 1805).
Two men successively
held the position of First Lord of the Admiralty in 1812: Charles
Philip Yorke (1764-1834) and Robert Dundas, 2nd Viscount Melville
(1771-1851).
1 Admiral of the Fleet: HRH the Duke of Clarence (1765-1837), appointed
1811. 11
Admirals (61)
21 Admirals of the Red: 20 Admirals of the White: James Hawkins Whitshed (1762-1849), 12
August 1812 20 Admirals of the Blue: Sir Robert Calder, Bt. (1745-1818), 31
July 1810
Vice-admirals (65)
22 Vice-admirals of the Red: Sir James de Saumerez, Bt. (1757-1836),
31 July 1810 19 Vice-admirals of the White: George Martin (1764-1847), 12 August
1812 24 Vice-admirals of the Blue: Francis Pickmore (1756-1818), 12
August 1812
Rear-admirals (64)
19 Rear-admirals of the Red: Sir Thomas Thompson, Bt. (1766-1828),
12 August 1812 17 Rear-admirals of the White: Thomas Francis Freemantle (1765-1819),
12 August 1812 24 Rear-admirals of the Blue: Sir Richard King, Bt. (1774-1834)
12 August 1812 12
In 1812, there were also 31 rear-admirals who were considered ‘superannuated’ ( incapacitated

CHALLENGER (history/fa7b9e6c_35157.txt, unjudged):
One of the earliest mentions of an English admiral is by Thoyras (1747, p.270) when he lists Richard de Lucy as an admiral in 1224. By 1294, as the English Navy expanded, Thoyras (1747, p.270) lists admirals with geographic areas of responsibility such as “Of the North”, “Of the South” and “Of the Weft” (or West). In 1360, Thoyras (1747, p.270) notes that one John Beauchamp effectively became admiral of all the English fleets, with this appointment being a predecessor to the future post of Admiral of the Fleet. From approximately the 1290s to 1400, admirals are listed as either “Admirals of the North” or “Admirals of the Weft”. (Thoyras, 1747, p.270-271). From 1406, Thoyras (1747, p.271) notes the posts are combined into the new title of “Admirals of England”. The first post-holder, John Beaufort, greatly expanded the jurisdiction of the Admiralty taking “…cognizance of criminal as well as civil matters”. (Thoyras, 1747, p.271). In 1345, John Holland purchased for himself and his son, a lifetime grant in the post of Admiral of England, Ireland and Aquitain, thereby again increasing the jurisdiction of the Admiralty. The first mention of Lord High Admiral is made in 1485 (Thoyras, 1747). Initially, the Admiral of England (later known as the Lord High-Admiral) was the only admiral with captains operating the ships and royal officers (aka nobility) commanding the ships via a commission from the monarch. “The office of Lord High Admiral dates from the 14th century, when the English Navy consolidated into one force. Originally responsible for aspects of Navy policy, the position

VERDICT: 

## 11. history / 68604_398 slot 9 delta -0.083

QUERY: Is there a formal ontology for documenting historical timelines?
<p>Is there a common process/system/platform that researchers use to document historical timelines?</p>
<p>For example:</p>
<p>If I wanted to track the history of some organization. I would want to track things like:</p>
<ul>
<li>Events</li>
<li>Locations</li>
<li>People</li>
<li>Groups</li>
<li>...Eras?</li>
</ul>
<p>Also, how those things intersect -- an Era that took place between Year X and Year Y contained a Event which took place at a Location and involved Person A, Person B, and Group C.</p>
<p>This is core content modelin

INCUMBENT (history/73b23d63_53716.txt, judged relevant):
Prosopography is the study of groups of people through the collective examination of their lives, often using historical records, literary sources, and artifacts. Rather than focusing on a single individual, prosopography analyzes patterns in a group, such as shared roles, relationships, or social status, to better understand historical societies and social structures. This method is especially useful when detailed information about individuals is scarce, as it allows researchers to draw insights from broader trends and connections between people. By assembling these collective biographies, prosopography helps reconstruct historical contexts and offers a richer understanding of past societies.

This resource introduces the concept of the prosopon, an invaluable tool in historical and literary studies that represents a collected literary depiction of a person. It encompasses all forms of information about an individual, gathered from texts, records, and artifacts. Often, our understanding of past people comes from these constructions, whether through others’ representations or physical records, which can be fragmented or biased. Despite these limitations, prosopons help us piece together comprehensive portraits of individuals, revealing their societal roles, personal relationships, and historical significance. The resource also explores the Theory of Identity, examining aspects like roles, titles, ethnicity, gender, and religion to understand the multi-dimensional identities of historical figures. Finally, it emphasizes the importance of scoping prosopography projects, guidi

CHALLENGER (history/85ae30dc_322360.txt, unjudged):
vteHistory of mathematics (timeline)By topic Algebra timeline Algorithms timeline Arithmetic timeline Calculus timeline Grandi's series Category theory timeline Topos theory Combinatorics Functions Logarithms Geometry Trigonometry timeline Group theory Representation theory Information theory timeline Logic timeline Math notation Number theory timeline Statistics timeline Probability Topology Manifolds timeline Separation axioms Numeral systems Prehistoric Ancient Hindu-Arabic By ancient cultures Mesopotamia Ancient Egypt Ancient Greece China Inca India Medieval Islamic world Controversies Brouwer Hilbert Over Cantor's theory Leibniz Newton Hobbes Wallis Other Women in mathematics timeline Approximations of π timeline Future of mathematics Category

VERDICT: 

## 12. history / 68862_380 slot 10 delta -0.136

QUERY: Could the &quot;Taffy&quot; Groups have been reinforced by air during the battle of Leyte Gulf?
<p>During the <a href="https://en.wikipedia.org/wiki/Battle_of_Leyte_Gulf#Crisis_%E2%80%93_U.S._Seventh_Fleet%27s_calls_for_help" rel="nofollow noreferrer">Battle for Leyte Gulf</a>, most elements of the American fleet fought two &quot;side&quot; groups of the Japanese fleet to the south and north of the main action, which would take place off Samar in the San Bernardino Strait. The end result was that only small task forces of escort carriers and destroyers, Taffy One, Two, and Three, stood between

INCUMBENT (history/3f8f6891_51645.txt, judged relevant):
situation looks almost hopeless and the group has requested air and surface aid from Admiral Stump. There is nothing more he can do. All of his available aircraft, 24 Avenger torpedo planes with Wildcat fighter protection, have been launched against the Japanese.

The destroyer screen of the beseiged carriers has been sent against the Jap force. This is a brave and desperate effort to turn, stop, or slow down the battleships and relieve the carriers from the merciless shelling.

The Kaiser carrier Midway (St. Lo) has been sunk. the number of survivors is unknown.

The destroyer screen of the group has been sent to attack. A lone destroyer, the USS Heermann has sent this message: "I am engaging enemy force consisting of two battleships, eight cruisers, twelve destroyers. Action continues."

The Heermann has been hit and is dead in the water.

The destroyers' gallant attack was in vain and all but two have been sunk or crippled.

0920 - The most remote of the carriers, Taffy One, is under air attack and cannot come to the aid of the other groups.

The Japanese force has flanked Taffy Three and is shelling the five carriers remaining who are now without either protection or ability to launch offensive aircraft. The Japanese force is, at the same time, rapidly overtaking Taffy Two whose flagship Natoma Bay is astern of the other carriers. From the ships of this group the salvoes of the battleships can be clearly seen as geysers in the water.

This battle is without the usual confusion of battle. Every seaman in the ships of Taffy Two can estimate that within 20 minutes, the she

CHALLENGER (history/5954a808_51685.txt, unjudged):
First, let's correct this... The Taffy units fought bravely, until they had run out of serviceable aircraft This is incorrect. Taffy did not run out of aircraft, Kurita turned away. After Kurita turned away, aircraft were still returning and air attack continued. Yamato and Chikuma were attacked by Avengers from Taffys 2 and 3.

VERDICT: 

## 13. history / 69881_319 slot 8 delta -0.074

QUERY: Did Hannibal make coins while campaigning using captured precious metals, Roman coins and perhaps bronze statues?
<p>The process of minting coins at the time would be similar to the Roman method and I assume it would have been normal for any large army to carry the required tools and craftsmen while campaigning to melt down captured material, coinage etc to generate his own coinage to pay the troops.  Rather than say pay the troops in coins looted from the Romans bearing the image of Roman emperors and gods.</p>
<blockquote>
<p>Metal was heated in a hot fire or furnace. It was either melted in

INCUMBENT (history/92db9b8c_43105.txt, judged relevant):
# NGC Ancients: The Coinage of Ancient Carthage, Part 4
Posted on 9/14/2021
Carthage continued to produce impressive coins in the decades before its destruction.
In the previous column we focused on Carthaginian coins from the city’s colonies in Spain. In this final installment we’ll cover the remaining coinage of the empire of ancient Carthage, from the Second Punic War (218-201 B.C.) until the sacking of the city in 146 B.C.
The Second Punic War sprung from Carthaginian discontent about the treaty terms and reparations due to Rome from the First Punic War (264-241 B.C.). This allowed for warlike leaders in Carthage to take command, the most famous being Hannibal.
The conflict was sparked in 219 B.C. when Hannibal conquered the pro-Roman city of Saguntum in Spain, causing the Romans to declare war. Hannibal surprised the Romans when he led his armies on a daring march to the Alps, crossing them in the peak of the winter of 218 B.C. and descending into the heart of Northern Italy. He went on to defeat army after army and was almost unbeatable during the next few years he was in Italy.
The Carthaginian coins of the Second Punic War were struck in many locations, which led to many different designs being used. We’ll review a few of the coins minted at Carthage itself and then look at coins minted in other locations by the Carthaginians or their allies.

CHALLENGER (history/7907f0d1_231380.txt, unjudged):
A gifted diplomat as well as a brilliant tactician, Hannibal s strategy depended on breaking up Rome s network of alliances with the Greek-speaking cities and rural Italic tribes of Italy. One powerful group was the Brettii, inhabiting the region of Calabria (the  toe  of the Italian  boot ). Hannibal issued coins to pay his troops in this region   including an electrum 3 8 shekel, that closely copied a contemporary Roman silver coin, and a silver quarter-shekel with the usual Carthaginian design of Tanit and a standing horse.

VERDICT: 

## 14. history / 70693_263 slot 10 delta -0.177

QUERY: How did Raphael paint halos?
<p>Yesterday, I had the pleasure of seeing <a href="https://en.wikipedia.org/wiki/Raphael_Cartoons" rel="nofollow noreferrer">Raphael's cartoon</a> &quot;Christ's Charge to Peter&quot;:</p>
<p><a href="https://i.sstatic.net/5kisJ.jpg" rel="nofollow noreferrer"><img src="https://i.sstatic.net/5kisJ.jpg" alt="Image of Raphael's &quot;Christ's Charge to Peter&quot;" /></a></p>
<p>In it, Christ and the apostles are depicted with halos that appear to be perfect <a href="https://en.wikipedia.org/wiki/Ellipse" rel="nofollow noreferrer">ellipses</a>.</p>
<p><a href="https:

INCUMBENT (history/11f7a286_36205.txt, judged relevant):
 Year C, 3rd Sunday of Easter


Feed my Lambs

In the year 1515 Raphael made this design on paper for a large tapestry for Pope Leo X. It depicts Jesus’ commission to Peter: “Feed my lambs.” This scene took place when Jesus appeared to his disciples for the third time after his resurrection.

Raphael combines the biblical narrative in John 21 with Matthew 16:18-19, where – some time before his resurrection – Jesus gives the keys of the kingdom of heaven to Peter. Raphael does this because Peter later betrayed Jesus and his position as leader had become insecure. By combining the two scenes we not only see Peter’s honour restored, but also the confirmation of his charge and special position as the rock on which Jesus will build his church.

Peter kneels. He makes himself small, conscious of his guilt. The other disciples do not seem to agree with what is happening here. Raphael emphasizes their rivalry. One of them even makes a gesture of wanting to stop Jesus. Others point to themselves. But Jesus pontifically points to Peter.

CHALLENGER (history/d613c7d0_350428.txt, unjudged):
Rectangular halos, such as the one in the Codex Egberti, were predominantly used in the Early Middle Ages in portraits of still-living persons who were still alive at the time, in order to nevertheless represent the divine grace inherent in them. However, there are also pictorial works that show such figures with a rounded gloriole or no gloriole at all.

VERDICT: 

## 15. history / 71924_189 slot 8 delta -0.107

QUERY: Did Diaochan really love L&#252; Bu?
<p>I am watching a series on Three kingdoms (2010), were it is shown that, <a href="https://en.wikipedia.org/wiki/Diaochan" rel="nofollow noreferrer">Diaochan</a> was really in love with <a href="https://en.wikipedia.org/wiki/L%C3%BC_Bu" rel="nofollow noreferrer">Lü Bu</a>, it was only <a href="https://en.wikipedia.org/wiki/Wang_Yun_(Han_dynasty)" rel="nofollow noreferrer">Wang Yun</a>, who schemed an act of proposing her daughter as a bride to Dong Zhou in order to have chaos between him and his son. So, I was confirming about this, as I can't find any pro

INCUMBENT (history/fb8258c5_25888.txt, judged relevant):
Chinese historical records indicate that Lü Bu had a secret affair with one of Dong Zhuo's courtesans and he constantly feared that Dong Zhuo would find out. This was one of the reasons why he betrayed and assassinated Dong Zhuo in May 192. However, the courtesan's name was not recorded. The name "Diaochan" first appeared in Sanguozhi Pinghua as the courtesy name of Lü Bu's wife. A later zaju Duoji (now lost) explained the name, which literally means "sable cicada", as derived from the sable tails and jade decorations in the shape of cicadas which adorned the hats of high-ranking officials in the Eastern Han dynasty.[1]

In folk tales
In earlier stories, Diaochan was described as Lü Bu's wife who had been accidentally separated from Lü Bu during the political disorder and was unaware of the conspiracy against Dong Zhuo. Later stories removed this background and had her more actively participating in the scheme. Narrations about Diaochan's ultimate fate appeared in Yuan dynasty zaju. The titles of these lost works suggest Diaochan, viewed as a femme fatale, was decapitated by Guan Yu after Lü Bu's death, which was inherited by some of later Ming or Qing operas featuring Diaochan.[1]

CHALLENGER (history/8fb729d8_200073.txt, unjudged):
When Dong Zhuo was killed and her mission completed, according to legends she finally married Lü Bu and stayed by his side up until his defeat in battle of Xiapi. As Romance of Three Kingdoms says, Lü Bu refused to follow the advice to go outside the city in order to rearrange the defense systems, because Diaochan was ill and he wanted to stay by her side. That cost him his life, as after the battle by joint forces of Liu Bei and Cao Cao, the latter executed treacherous general, who betrayed his previous masters to have the best horse and the most beautiful woman, despite the fact that Lü Bu was strongest among the warriors. Diaochan wanted to die together with her husband, but was drawn away by Cao Cao followers.

VERDICT: 

## 16. history / 72360_164 slot 10 delta -0.113

QUERY: Was reincarnation part of the early Christian church?
<p>I am interested in learning more about whether reincarnation was a part of early Christian church dogma, and how that belief may have evolved or survived as the church splintered into multiple sects. The internet is a bit of a quagmire, divided mostly into three major categories: Catholic writings that regurgitate the modern church's stance, random new-ageish writings that have bad or no references, and some academic writings which I (as an aspiring fiction writer, not a scholar) cannot tell if they are disproven or seconded by other aut

INCUMBENT (history/81401c1f_21854.txt, judged relevant):
This is the third in a series of replies to David Knott’s response to my article on the Big Bang. (There is no need to have read any of this to understand what follows, but for part 1, on the relevance of the Big Bang debate to Christianity, click here, and part 2, Reflections on Genesis chapter 1, click here.)

My interest, here and in the following article, is to discuss Knott’s statements about the soul and the creation of humans. He says: “The Biblical account of man’s creation implies the physical came first, then the spiritual”. As evidence he cites Genesis 2.7: “Then the Lord God formed a man from the dust of the ground and breathed into his nostrils the breath of life, and the man became a living being”, then says: “The Hebrew word for ‘being’ here can also mean soul. I do not see here, or anywhere else in the Bible the notion of souls existing before their physical bodies”.

The doctrine that Knott is trying to deny here is that of pre-existence, and by implication reincarnation. My immediate response is that, if there is no evidence for pre-existence or reincarnation in the Bible, that would be a sad reflection on it for being so lacking. However, is what Knott says actually true? He is subscribing to the modern official Christian doctrine that the soul is born with the body. But how credible is this?

In what follows I will argue that early Christians believed in pre-existence and reincarnation, that there is evidence for this in the Bible, and that these ideas were declared a heresy as late as 553 CE. This was not done by the Church, rather by the Emperor Justin

CHALLENGER (history/5084bc82_22121.txt, unjudged):
Reincarnation(Excerpt Chapter 4, 'The Gnostics') "In the Pagan Mysteries it was believed that a soul progresses towards the realization of the Gnosis over many lifetimes.  The Pagan initiate Plutarch explains that the unenlightened soul is attracted back into physical incarnation over and over again by force of habit: 'We know that the soul is indestructible and should think of its experience as like that of a bird in a cage. If it has been kept in the body for a long time and become tamed to this life as a result of all sorts of involvements and long habituation, it will alight again back to a body again birth after birth and will never stop or give up becoming entangled in the passions and chances of this world.'  Although it was eventually exorcised from mainstream Christianity, this Pagan idea was embraced by the early Gnostic Christians. The Gnostic sage Basilides taught that Gnosis was the consummation of many lives of effort. 'The Secret Book of John' teaches that a soul will continue to reincarnate until it is eventually 'saved from its lack of perception, attains Gnosis, and so is perfected', after which 'it no longer goes into another flesh'. The 'Pistis Sophia' teaches that a soul cannot be brought into the Light until, through many lifetimes of experience, it has understood all of the Mysteries. Having progressed on the spiritual journey during this life, however, its next incarnation will be into a 'righteous body which shall find the God of Truth and the Higher Mysteries'. Plato tells us that the dead have the choice of drinking from the 'Spring of Memory' and

VERDICT: 

## 17. history / 76952_71 slot 8 delta -0.123

QUERY: Why did German typewriters not use Fraktur?
<p><a href="https://en.wikipedia.org/wiki/Fraktur" rel="noreferrer">Fraktur</a> was the dominant typeface in Germany for centuries until the Nazis suddenly stopped using it in WWII.</p>
<p>I read <sup>citation needed</sup> that the Nazis, before they suddenly marked Fraktur as &quot;Jewish&quot; in 1941, tried to introduce Fraktur typewriters in 1933 but quickly gave up.</p>
<p>That inspired my questions: Why were German typewriters not using Fraktur in the first place?</p>
<p>The Nazis stopped using Fraktur in 1941, officially because they marked it

INCUMBENT (history/43e4284a_9573.txt, judged relevant):
Ironically, Fraktur’s downfall came not from foreign influence but from within. In 1941, the Nazi regime abruptly banned Fraktur, labeling it “Schwabacher Judenlettern” (Schwabacher Jewish letters) — a wildly inaccurate claim likely rooted in convenience and propaganda. The shift aimed to align with international modernism and make German writing more legible abroad. This sudden rejection, after centuries of use, left Fraktur tainted by association and hastened its decline in everyday life.


Fraktur Today: Revival and Reclamation
In recent decades, Fraktur has experienced a cautious but creative resurgence. Type designers, calligraphers, and artists are reinterpreting the style — either leaning into its expressive potential or subverting its historical baggage. You’ll see Fraktur-inspired type in music branding, beer labels, fashion, and streetwear — anywhere a sense of heritage or drama is called for. Crucially, modern use often strips away the nationalist overtones and reclaims Fraktur as part of a wider, global design language.

CHALLENGER (history/f21a6fcf_144669.txt, unjudged):
From the late 18th century to the late 19th century, Fraktur was progressively replaced by Antiqua) as a symbol of the classicist age and emerging cosmopolitanism in most of the countries in Europe that had previously used Fraktur. This move was hotly debated in Germany, a controversy known as the Antiqua-Fraktur dispute . The shift affected mostly scientific writing in Germany, whereas most belletristic literature and newspapers continued to be printed in Fraktur.

VERDICT: 

## 18. history / 77003_69 slot 8 delta -0.148

QUERY: What do the writings of the NKVD general Lyushkov after his defection to Japan contain?
<p>I was recently looking at a paper about Lyushkov, an NKVD general who defected to Japan after he was recalled to Moscow where he would have been likely arrested and eventually shot.
Several Japanese who came into contact with him described him as &quot;very bright&quot; and &quot;very intelligent&quot; among other things and according to a paper about him he became a prolific writer &quot;writing 40 pages a day&quot;.
From his writings very little remains today including these two papers which are inacce

INCUMBENT (history/6bf1c8ab_9218.txt, judged relevant):
 Though he knowingly endangered family, friends, and colleagues, Lyushkov dared to disobey, fleeing to Japan in June 1938. His defection contributed to destroying Yezhov and Marshal Blyukher, and causing border war with Japan at Khasan/Changkufeng. Till 1945 Lyushkov worked for the Japanese Army, which eventually murdered him at Dairen. He had survived seven years but ultimately failed to weaken Stalin, who outlasted him handily.

CHALLENGER (history/280d1eac_144235.txt, unjudged):
Lyushkov was able to detail strength of the Red Army in the Far East, Siberia and Ukraine, simultaneously providing Soviet military radio codes. He was considered highly intelligent and dedicated, producing great volumes of written material, but there was some uncertainty about his ability to provide useful information specific to military operations.

VERDICT: 

## 19. history / 77154_56 slot 8 delta -0.148

QUERY: Why did the Civil Service in the UK become so associated with Oxbridge?
<p>In the UK there is definitely a perception that the Civil Service is the preserve of privileged Oxbridge graduates, possibly even only those who came from public schools (Eton, Winchester, etc). It is debatable how true this perception is in the modern day, but certainly it was felt strongly enough in the past to be satirised in television (most prominently in, say, Yes Minister and its sequel).</p>
<p>There is the definite feeling that the Civil Service is or was a sort of old boys' club, where those with connections w

INCUMBENT (history/3551ddfe_7116.txt, judged relevant):
The Northcote–Trevelyan Report recommended a permanent, unified, politically neutral civil service, with appointments made on merit, and a clear division between staff responsible for routine ("mechanical") and those engaged in policy formulation and implementation ("administrative") work. The report was not implemented, but it came as the bureaucratic chaos in the Crimean War demonstrated that the military was as backward as the civil service. A Civil Service Commission was set up in 1855 to oversee open recruitment and end patronage as Parliament passed an Act "to relieve the East India Company from the obligation to maintain the College at Haileybury".[13] Prime Minister Gladstone took the decisive step in 1870 with his Order in Council to implement the Northcote-Trevelyan proposals.[14] This system was broadly endorsed by commissions chaired by Playfair (1874), Ridley (1886), MacDonnell (1914), Tomlin (1931) and Priestley (1955).

The Northcote–Trevelyan model remained essentially stable for a hundred years. This was attributed to its success in removing corruption, delivering public services, even under stress of war, and responding effectively to political change. Patrick Diamond argues:

The Northcote-Trevelyan model was characterised by a hierarchical mode of Weberian bureaucracy; neutral, permanent and anonymous officials motivated by the public interest; and a willingness to administer policies ultimately determined by ministers. This bequeathed a set of theories, institutions and practices to subsequent generations of administrators in the central state.[15]

The

CHALLENGER (history/4f4fc948_141217.txt, unjudged):
The Civil Service, along with  arm s length  public bodies, are responsible for enacting government policy and overseeing a wide range of sectors across British society. Here there is a consistent picture of over-representation of those from elite educational backgrounds. Civil service permanent secretaries (59 ), Foreign Office diplomats (52 ), and Public Body Chairs (45 ) have among the highest rates of independently educated in their ranks.

VERDICT: 

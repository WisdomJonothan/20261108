# Judge disagreement review (top 20 by score gap)
Direct = Qwen3-4B at 112 words; Llama = Llama-3.1-8B at 112 words.

## 1. genealogy / 11739_303 — gap 0.1934
Query: How far back is a mtDNA match of genetic distance 2 likely to be?
<p>On FamilyTreeDNA, for haplogroup H1, my closest matches are all genetic distance of 2 for HVR1, HVR2, CODING REGIONS.  I don't see any tool to predict the number of generations like we have for yDNA test, so I'm wondering what is the likely number of generations given that mtDNA mutates more slowly than yDNA?</p>

- Direct: slot 10 delta +0.000 (query delta 0.0)
- Llama:  slot 8 delta -0.193 (query delta -0.1934)

Pair slot 8: incumbent genealogy/733d4d68_6994.txt (rel 1), challenger genealogy/11739_9ab1f58b8f954ba5_17.txt (rel 0), true delta +0.000

INCUMBENT: When we looked at the Y-DNA STR tests we looked directly at the number of repeats at STR markers or the identity of bases at particular locations. mtDNA data analysis is different. Here we compare how each individual differs from reference standards. The first produced was the Cambridge Reference Standard (CRS), now superseded by the corrected revised Cambridge Reference Standard (rCRS), based on a European who had haplogroup H. A second standard, the Reconstructed Sapiens Reference Sequence (RSRS), was produced more recently and was an attempt to compare mtDNA against a reference with an olde

CHALLENGER: I am about to add a mito test to my Family Tree DNA portfolio currently consisting of a Y-37 and autosomal (Family Finder). I have already determined my haplogroup (T2a1b1a1) through NatGeo’s Geno 2.0 who uses Helix Labs as their testing source. They, however, refuse to provide further testing details for use on MitoSearch, etc. Should I spend the extra $100 for fms? Also, you said your hvr1&hvr2 match with “f” missing from haplo was 2000 years ago when split from its parent group “plus or minus 3000 years”. Then you state your in-common ancestor must have been 2000-5000 years old. But wouldn’

Pair slot 9: incumbent genealogy/0f7f8e8f_7116.txt (rel 0), challenger genealogy/11739_9ab1f58b8f954ba5_17.txt (rel 0), true delta +0.000

INCUMBENT: For recent genealogy, exact matching on the mtDNA full sequence is used to confirm a common ancestor on the direct maternal line between two suspected relatives. Because mtDNA mutations are very rare, a nearly perfect match is not usually considered relevant to the most recent 1 to 16 generations. In cultures lacking matrilineal surnames to pass down, neither relative above is likely to have as many generations of ancestors in their matrilineal information table as in the above patrilineal or Y-DNA case: for further information on this difficulty in traditional genealogy, due to lack of matril

CHALLENGER: I am about to add a mito test to my Family Tree DNA portfolio currently consisting of a Y-37 and autosomal (Family Finder). I have already determined my haplogroup (T2a1b1a1) through NatGeo’s Geno 2.0 who uses Helix Labs as their testing source. They, however, refuse to provide further testing details for use on MitoSearch, etc. Should I spend the extra $100 for fms? Also, you said your hvr1&hvr2 match with “f” missing from haplo was 2000 years ago when split from its parent group “plus or minus 3000 years”. Then you state your in-common ancestor must have been 2000-5000 years old. But wouldn’

## 2. history / 70693_263 — gap 0.1772
Query: How did Raphael paint halos?
<p>Yesterday, I had the pleasure of seeing <a href="https://en.wikipedia.org/wiki/Raphael_Cartoons" rel="nofollow noreferrer">Raphael's cartoon</a> &quot;Christ's Charge to Peter&quot;:</p>
<p><a href="https://i.sstatic.net/5kisJ.jpg" rel="nofollow noreferrer"><img src="https://i.sstatic.net/5kisJ.jpg" alt="Image of Raphael's &quot;Christ's Charge to Peter&quot;" /></a></p>
<p>In it, Christ and the apostles are depicted with halos that appear to be perfect <a href="https://en.wikipedia.org/wiki/Ellipse" rel="nofollow noreferrer">ellipses</a>.</p>
<p><a href="https://i.sstatic.net/gZzD6.png" rel="nofollow noreferrer"><img src="https://i.sstatic.net/gZzD6.png" alt="Enlarged image of halo" /></a></p>
<p>I only know of one method to draw an ellipse (<a href="https://www.youtube.com/watch?v=JvciWGTdBq8" rel="nofollow noreferrer">involving two pins and a loop of string</a>), but a naive application of this method punctures the paper with the pins. So how did Raphael paint his halos?</p>

- Direct: slot 10 delta -0.177 (query delta -0.1772)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/bebf9f97_210407.txt (rel 0), challenger history/2073bd88_350463.txt (rel 0), true delta +0.000

INCUMBENT: The Pope also commissioned Raphael to design ten tapestries to hang on the walls of the Sistine Chapel. Raphael managed to complete seven cartoons (full sized preparatory drawings), which were sent to be woven in the workshop of Flemish weaver Pieter Coecke van Aelst. They were hung in the chapel shortly before Raphael's death.

CHALLENGER: Rectangular halos , such as the one in the Codex Egberti , were predominantly used in the Early Middle Ages in portraits of still-living persons who were still alive at the time, in order to nevertheless represent the divine grace inherent in them. However, there are also pictorial works that show such figures with a rounded gloriole or no gloriole at all.

Pair slot 9: incumbent history/45fed4d7_210358.txt (rel 0), challenger history/2073bd88_350463.txt (rel 0), true delta +0.000

INCUMBENT: 1510Disputation of the Holy SacramentThis fresco in the Stanza della Segnatura, one of the four Raphael Rooms in the Apostolic Palace in the Vatican, is one of four paintings in the room which depict separately: philosophy, poetry, theology, and law. The Disputation of the Holy Sacrament represents theology and shows the occupants of a Catholic Church underneath the span of heaven above their sacred altar. The fresco represents Christianity's victory over Philosophy, which is depicted in The School of Athens, the fresco on the opposite wall. In heaven we see Christ in the center with the Virgi

CHALLENGER: Rectangular halos , such as the one in the Codex Egberti , were predominantly used in the Early Middle Ages in portraits of still-living persons who were still alive at the time, in order to nevertheless represent the divine grace inherent in them. However, there are also pictorial works that show such figures with a rounded gloriole or no gloriole at all.

## 3. bitcoin / 124603_8 — gap 0.148
Query: Question about bitcoin mining difficulty adjustment formula in book mastering bitcoin 3rd edtion
<p>In book &quot;Mastering bitcoin&quot; 3rd, page 278, There is a formula about bitcoin mining difficulty adjustment formula as below:</p>
<pre><code>        New Target = Old Target * (20,160 minutes / Actual Time of Last 2015 Blocks)
</code></pre>
<p>I think it should be:</p>
<pre><code>        New Target = Old Target * (Actual Time of Last 2015 Blocks / 20,160 minutes)
</code></pre>
<p>Anyone has comments about it? Thanks.</p>

- Direct: slot 8 delta -0.148 (query delta -0.148)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent bitcoin/6c9cebd4_1912.txt (rel 1), challenger bitcoin/cfad6f43_143777.txt (rel 0), true delta +0.000

INCUMBENT: Target Block Time
Difficulty is tweaked to ensure blocks are mined every 10 minutes, regardless of miner activity.

Adjustment Formula
The difficulty is recalculated using a formula that compares the actual mining time to the expected time.

Network Security
Consistent difficulty ensures Bitcoin’s security by preventing too-fast or too-slow block production. Blockchain development services can assist in understanding and optimizing these processes.

CHALLENGER: 1 . What is Difficulty Adjustment in Bitcoin? 2 . Importance of Mining Difficulty Adjustment in Bitcoin 3 . How Bitcoin Maintains Mining Difficulty? 4 . Regular Adjustments 5 . Hash Rate Monitoring 6 . Target Block Time 7 . Adjustment Formula 8 . Network Security 9 . Factors Influence Bitcoin's Difficulty Adjustment 10 . What Happens If Mining Difficulty is Not Adjusted Properly? 11 . Are there Limits to Difficulty Adjustments? 12 . Adjustment Limits 13 . Minimum Adjustment Interval 14 . Target Block Time 15 . Network Stability 16 . Can Difficulty Changes Affect Bitcoin's Value? 17 . How Does 

Pair slot 9: incumbent bitcoin/6bcc83a8_10627.txt (rel 0), challenger bitcoin/cfad6f43_143777.txt (rel 0), true delta +0.000

INCUMBENT: The target difficulty represents how difficult it is to mine a given block. This difficulty is not fixed and needs to constantly adjust to changing network hash rates. The difficulty adjustment MUST be calculated using a linear-weighted moving average (LWMA) algorithm (2) $$ \newcommand{\solvetime}{ \mathrm{ST_i} } \newcommand{\solvetimemax}{ \mathrm{ST_{max}} } $$ $$ \begin{align} & \textit{weighted_solve_time} = \sum\limits_{i=1}^N(\solvetime*i) \\ & \textit{weighted_target_time} = (\sum\limits_{i=1}^Ni) * \mathrm{T} \\ & \textit{difficulty} = \mathrm{D_{avg}} * \frac{\textit{weighted_target

CHALLENGER: 1 . What is Difficulty Adjustment in Bitcoin? 2 . Importance of Mining Difficulty Adjustment in Bitcoin 3 . How Bitcoin Maintains Mining Difficulty? 4 . Regular Adjustments 5 . Hash Rate Monitoring 6 . Target Block Time 7 . Adjustment Formula 8 . Network Security 9 . Factors Influence Bitcoin's Difficulty Adjustment 10 . What Happens If Mining Difficulty is Not Adjusted Properly? 11 . Are there Limits to Difficulty Adjustments? 12 . Adjustment Limits 13 . Minimum Adjustment Interval 14 . Target Block Time 15 . Network Stability 16 . Can Difficulty Changes Affect Bitcoin's Value? 17 . How Does 

## 4. history / 77154_56 — gap 0.148
Query: Why did the Civil Service in the UK become so associated with Oxbridge?
<p>In the UK there is definitely a perception that the Civil Service is the preserve of privileged Oxbridge graduates, possibly even only those who came from public schools (Eton, Winchester, etc). It is debatable how true this perception is in the modern day, but certainly it was felt strongly enough in the past to be satirised in television (most prominently in, say, Yes Minister and its sequel).</p>
<p>There is the definite feeling that the Civil Service is or was a sort of old boys' club, where those with connections would head to after finishing their classics degrees at Oxbridge. Attending Oxford in the 2010s, the idea of the Civil Service being a natural next destination was certainly still prevalent.</p>
<p>I contrast this to the equivalent in the US: as far as I can tell there is no association between the US CS and, say, the Ivy League. The perception in the US seems to be the polar opposite, with almost a sense of disdain by Ivy League graduates of the idea of working as a bureaucrat as opposed to entering the corporate world.</p>
<p>What were the historic factors that led to this perception in the UK, and to what degree was it ever true?</p>

- Direct: slot 8 delta -0.148 (query delta -0.148)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/3551ddfe_7116.txt (rel 1), challenger history/4f4fc948_141217.txt (rel 0), true delta +0.000

INCUMBENT: The Northcote–Trevelyan Report recommended a permanent, unified, politically neutral civil service, with appointments made on merit, and a clear division between staff responsible for routine ("mechanical") and those engaged in policy formulation and implementation ("administrative") work. The report was not implemented, but it came as the bureaucratic chaos in the Crimean War demonstrated that the military was as backward as the civil service. A Civil Service Commission was set up in 1855 to oversee open recruitment and end patronage as Parliament passed an Act "to relieve the East India Comp

CHALLENGER: The Civil Service, along with  arm s length  public bodies, are responsible for enacting government policy and overseeing a wide range of sectors across British society. Here there is a consistent picture of over-representation of those from elite educational backgrounds. Civil service permanent secretaries (59 ), Foreign Office diplomats (52 ), and Public Body Chairs (45 ) have among the highest rates of independently educated in their ranks.

Pair slot 9: incumbent history/6d785e27_141191.txt (rel 0), challenger history/4f4fc948_141217.txt (rel 0), true delta +0.000

INCUMBENT: Politics, the media, and public service all show high proportions of privately educated in their number, including 65  of senior judges, 59  of civil service permanent secretaries and 57  of the House of Lords. One third of regular newspaper columnists attended both private school and Oxbridge. Sport (particularly football), the arts and local government were areas with the lowest numbers of those coming from elite educational institutions. Numbers of Oxbridge graduates reached as high as 71  of senior judges and 57  of Theresa May's cabinet.

CHALLENGER: The Civil Service, along with  arm s length  public bodies, are responsible for enacting government policy and overseeing a wide range of sectors across British society. Here there is a consistent picture of over-representation of those from elite educational backgrounds. Civil service permanent secretaries (59 ), Foreign Office diplomats (52 ), and Public Body Chairs (45 ) have among the highest rates of independently educated in their ranks.

## 5. cardano / 6918_119 — gap 0.1413
Query: How does Cardano execute real time transactions if block time is 20 seconds?
<p>New to the crypto space, please correct me if I misunderstood the meaning of block time.</p>
<p>When a cardano node submits a transaction, it goes to a local mempool and submitted on the blockchain for verification. Let's assume this part is instantaneous from the wallet node.</p>
<p><a href="https://developers.cardano.org/docs/stake-pool-course/introduction-to-cardano/#slots-and-epochs" rel="nofollow noreferrer">The official docs</a> state that blocks are produced every 20 seconds. In other words, even if the validator nodes receive the transaction immediately, they will wait upto another 20 seconds, and then create a new block to broadcast.</p>
<p>This implies even a low assurance for block will take atleast 20 seconds. In my experience with cardano wallets (before the dexes were launched), it rarely took more than 10 seconds to get a high assurance for transaction.</p>
<p>I also read on the forums that 20 seconds block time is configurable, and can't be enforced on a node. So are we relying on SPOs to use a low block creation time interval to solve real time transactions in the long run?</p>

- Direct: slot 9 delta +0.141 (query delta 0.1413)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent cardano/7235_31b1d15a10971633_2.txt (rel 0), challenger cardano/3cb01889_3261.txt (rel 0), true delta +0.000

INCUMBENT: When you submit a transaction, you submit it, either directly yourself or through some public API like blockfrost, to a Cardano node. This node validates that it is correct, inserts it into its own “mempool” (a list of transactions ready to be included in a block), and then s it with each node it is connected to. Each of those nodes validates the transaction, inserts it into its mempool, and then propagates it downstream to *their* peers. Through this process, a transaction is “gossiped” around the network. In practice, a given transaction reaches nearly every node in just a few seconds. This 

CHALLENGER: Cardano is a third-generation blockchain platform named after Ada Lovelace, the 19th-century mathematician recognized as the first computer programmer. It was launched in September 2017 by IOHK (Input Output Hong Kong), a technology company founded by Charles Hoskinson and Jeremy Wood. Cardano seeks to offer a more balanced and sustainable ecosystem for complex deal settlement, DApp development, and interoperability with other systems seeking integration. The native coin ADA powers the Cardano blockchain and serves as a means of payment for transaction fees and as a medium of exchange. Ourobor

Pair slot 9: incumbent cardano/45863ee3_0059.txt (rel 0), challenger cardano/3cb01889_3261.txt (rel 0), true delta +0.000

INCUMBENT:  In Cardano, time is divided into epochs and slots, with each slot lasting one second and offering an opportunity for a block to be produced. This precise timekeeping mechanism is crucial for the operation of Cardano's Ouroboros consensus protocol and helps ensure the network's efficiency and security.

CHALLENGER: Cardano is a third-generation blockchain platform named after Ada Lovelace, the 19th-century mathematician recognized as the first computer programmer. It was launched in September 2017 by IOHK (Input Output Hong Kong), a technology company founded by Charles Hoskinson and Jeremy Wood. Cardano seeks to offer a more balanced and sustainable ecosystem for complex deal settlement, DApp development, and interoperability with other systems seeking integration. The native coin ADA powers the Cardano blockchain and serves as a means of payment for transaction fees and as a medium of exchange. Ourobor

## 6. history / 59006_644 — gap 0.1413
Query: How many naval mines were cleared for D-Day?
<p>I was interested to read in <a href="https://history.stackexchange.com/questions/40551/right-before-d-day-how-were-naval-mines-cleared-secretly">this question</a> about the clearing of naval mines for D-Day in 1944. There are estimates of the frequency of the <a href="https://www.junobeach.org/beach-obstacles/" rel="nofollow noreferrer">mines on the beach</a> -- <a href="https://www.history.com/news/d-day-normandy-wwii-facts" rel="nofollow noreferrer">history.com says</a> there were "4 million landmines along Normandy’s beaches", but I've not found anything on the number of mines cleared in the water.</p>

<p><strong>Does any source indicate or suggest how many naval mines were cleared by the nighttime minesweeping operation?</strong></p>

<p><a href="https://i.sstatic.net/zD7Rp.jpg" rel="nofollow noreferrer"><img src="https://i.sstatic.net/zD7Rp.jpg" alt="Mine targeting at sea"></a></p>

<p><a href="https://i.sstatic.net/iwuJn.png" rel="nofollow noreferrer"><img src="https://i.sstatic.net/iwuJn.png" alt="Minesweeping map"></a></p>

- Direct: slot 9 delta +0.141 (query delta 0.1413)
- Llama:  slot 8 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent history/4c1d0718_92326.txt (rel 1), challenger history/b4478409_92325.txt (rel 1), true delta +0.000

INCUMBENT: By June 1944, over 6,000 mines protected the approaches to Normandy beaches. Some advanced mines had ship counters that let several vessels pass before exploding, making minesweeping operations even more dangerous and unpredictable.



CHALLENGER: After sweeping the fire support areas, the ships moved to sweep as much of the transport anchorages as they could before they filled up with landing craft. Around 0330 on June 6, Squadron “A” had completed the assault mission and was watching the bombardment ships file into the fire support area. Their first mission accomplished, they anchored in a defensive line, called the Mason line, on the western edge of the Utah assault area to protect against small German surface craft based out of Cherbourg. Unlike many of the ships involved in Operation Neptune, the minesweepers’ job was not over afte

Pair slot 9: incumbent history/e25c9484_331347.txt (rel 0), challenger history/b4478409_92325.txt (rel 1), true delta +0.000

INCUMBENT: On June 5-6, 1944, Allied forces kicked off complex minesweeping operations, using organized flotillas to clear ten specific approach channels to the Normandy beaches. Minesweepers worked with U.S. destroyers and British destroyers to break through the German mine barrier before the main invasion fleet showed up.

CHALLENGER: After sweeping the fire support areas, the ships moved to sweep as much of the transport anchorages as they could before they filled up with landing craft. Around 0330 on June 6, Squadron “A” had completed the assault mission and was watching the bombardment ships file into the fire support area. Their first mission accomplished, they anchored in a defensive line, called the Mason line, on the western edge of the Utah assault area to protect against small German surface craft based out of Cherbourg. Unlike many of the ships involved in Operation Neptune, the minesweepers’ job was not over afte

## 7. travel / 198308_1103 — gap 0.1413
Query: What do from: and to: mean on a Schengen visa?
<p>A Schengen visa has from: and to: dates and duration 90 days.</p>
<p>A person has such a visa, which totals to 102 days, from: until to: dates.</p>
<p>I seem to have read that from: and to: limit the entry of the person, not the visa. The duration is still 90 days. So entering on the day of to: still gives you 90 days. Which means you need a stamp in the passport, to prove you have not overstayed.</p>
<p>Is this how it works or have I missunderstood?</p>

- Direct: slot 9 delta +0.141 (query delta 0.1413)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent travel/8edf9d69_7434.txt (rel 0), challenger travel/c53af577_2906.txt (rel 1), true delta +0.000

INCUMBENT: So, what is the 90/180 day rule? The 90/180 day rule means that visa-free travel to the Schengen area for short-term visits as well as travel under a multiple-entry visa valid for at least six months and with 90 days of duration of stay should not spend more than 90 days within 180 days in the Schengen territory. The count begins as soon as a traveller enters the Schengen Area and ends on the day they depart, and does not reset. Many travellers can get confused by this rule, which leads to a wrong calculation of how long they have stayed in Europe, and how long they are permitted to remain and

CHALLENGER: Short-stay Schengen Visas are only valid for a maximum of 3 months (90 days) within a six-month period.

Although multiple-entry visas are valid for longer periods - six months to five years - you may only spend 90 days (duration of stay on your visa sticker) in the Schengen Area within a rolling 180-day period (known as its validity period).

The validity period of your visa indicates how long you can use your visa. The dates are mentioned on your visa sticker. It will have a ‘From’ date, from which you are permitted to enter the Schengen Area, and an ‘Until’ date - the date on which you must

Pair slot 9: incumbent travel/60cdbcc5_1372.txt (rel 0), challenger travel/c53af577_2906.txt (rel 1), true delta +0.000

INCUMBENT: The Schengen 90-180 rule regulates short-term stays, limiiting travel to 90 days in any rolling 180-day period. Any travel to the Schengen Area by a non-EU/EEA national under 90 days is considered a short-term stay. If you want to spend more than 90 days, then you'll have to get a long-term visa or a residency permit. The day you enter Schengen counts as day 1, and entry/exit days count. The whole Schengen Area is treated as one. Changing countries doesn’t reset the counter. You must be compliant on every day of your trip (look back 180 days on each day of stay).

CHALLENGER: Short-stay Schengen Visas are only valid for a maximum of 3 months (90 days) within a six-month period.

Although multiple-entry visas are valid for longer periods - six months to five years - you may only spend 90 days (duration of stay on your visa sticker) in the Schengen Area within a rolling 180-day period (known as its validity period).

The validity period of your visa indicates how long you can use your visa. The dates are mentioned on your visa sticker. It will have a ‘From’ date, from which you are permitted to enter the Schengen Area, and an ‘Until’ date - the date on which you must

## 8. history / 55888_782 — gap 0.1357
Query: Why were the British Union of Fascist outfits so bad compared to the Nazis?
<p>Looking at Oswald Mosley's fascists "blackshirts", it struck me that the tailoring on their costumes was quite bad compared to their German counterparts.</p>

<p>Photos I've seen look like black boiler suits with a gold belt.</p>

<p>Now I know German engineering was supposed to be superior at that time. But was this also true for tailoring? </p>

<p>The Italians are known for good tailoring, so did the Nazis get their uniforms tailored by the Italian tailors who were governed by Mussolini at the time?</p>

<p>Was this simply a lack of funds on the part of the BUF? And if Oswald Mosely was friends with Hitler, why did the Nazis not help in providing better uniforms?</p>

<p>Or was there some other reason, for example trying to not look too ostentatious.</p>

<p>Was this a reason that the blackshirts were not taken very seriously because their uniforms looked so bad?</p>

- Direct: none (query delta 0.0)
- Llama:  slot 10 delta +0.136 (query delta 0.1357)

Pair slot 8: incumbent history/aa828e68_109430.txt (rel 1), challenger history/e8517c62_109631.txt (rel 0), true delta +0.000

INCUMBENT: However, the crisis in British society, which Mosley expected to capitalize on, failed to materialize in the 1930s. The factors which fed fascism on the continent—chronic unemployment, the threat of communism, and national grievances—were not as extreme in Britain. The BUF failed to create a nation-wide, mass movement. It suffered a set-back after the Olympia meeting in June 1934 when unnecessarily strong action was taken by the Blackshirts to silence hecklers. Its link with political violence and its increasingly anti-Semitic stance alienated moderate opinion, including Rothermere.


CHALLENGER: “Only one thing could have broken our movement — if the adversary had understood its principle and from the first day had smashed with extreme brutality the nucleus of our new movement.” —Adolf Hitler: to Nuremberg Nazi Party congress, 1933. Anarchism finds its antithesis in fascism; an understanding of its origins, practices and dynamics is essential for anarchists, as it holds up a mirror to our vision of social relations, magnifying everything we wish to do away with. It is also necessary to oppose it with everything we have, or it will do away with us. Only when we have permanently elimina

Pair slot 9: incumbent history/3b7195cc_351122.txt (rel 0), challenger history/e8517c62_109631.txt (rel 0), true delta +0.000

INCUMBENT: Mosley had a clutch of thuggish bodyguards known as the  Biff Boys,  whose ranks expanded and whose aggression increased as they became the Blackshirts, enforcers who mirrored Mussolini's national militia of the same name . They also executed the right-arm-extended salute used by the Italian Fascists and German Nazis . Moreover, the BUF staged dramatic rallies, where Mosley, a skilled speaker, demonstrated impassioned oratory in the vein of Mussolini and Adolf Hitler , with whom he developed collaborative relationships and whose governments provided financial support for the BUF. (In 1936 Hitl

CHALLENGER: “Only one thing could have broken our movement — if the adversary had understood its principle and from the first day had smashed with extreme brutality the nucleus of our new movement.” —Adolf Hitler: to Nuremberg Nazi Party congress, 1933. Anarchism finds its antithesis in fascism; an understanding of its origins, practices and dynamics is essential for anarchists, as it holds up a mirror to our vision of social relations, magnifying everything we wish to do away with. It is also necessary to oppose it with everything we have, or it will do away with us. Only when we have permanently elimina

## 9. history / 68862_380 — gap 0.1357
Query: Could the &quot;Taffy&quot; Groups have been reinforced by air during the battle of Leyte Gulf?
<p>During the <a href="https://en.wikipedia.org/wiki/Battle_of_Leyte_Gulf#Crisis_%E2%80%93_U.S._Seventh_Fleet%27s_calls_for_help" rel="nofollow noreferrer">Battle for Leyte Gulf</a>, most elements of the American fleet fought two &quot;side&quot; groups of the Japanese fleet to the south and north of the main action, which would take place off Samar in the San Bernardino Strait. The end result was that only small task forces of escort carriers and destroyers, Taffy One, Two, and Three, stood between the main Japanese fleet and American transports landing troops and supplies.</p>
<p>The Taffy units fought bravely, until they had run out of serviceable aircraft, and the much superior Japanese fleet retreated. But suppose the battle had continued. Halsey's Third Fleet, by this time, was too far away to help directly. But was it possible for it to fly in replacements for the aircraft (and bombs) lost by the Taffys in the earlier fighting? That is, after a one way trip, could they have refueled and continued the fight from the same escort carriers, but using new planes and pilots?</p>
<p><strong>Edit:</strong> My understanding is that Halsey's fleet was too far away for its planes to conduct a &quot;round trip&quot; attack on Kurita's fleet. My hypothesis was that Halsey could fly in enough planes on &quot;one way&quot; trips to replace the Taffy losses, maintaining their complement constant. Those planes would refuel on the Taffy carriers, and then make their strikes. Is this logistically possible? This would apply more to Taffy I and Taffy II, whose planes were operating at a longer distance, and perhaps not to Taffy III, which was already engaged with Japanese surface forces and on the verge of destruction. –</p>

- Direct: slot 10 delta -0.136 (query delta -0.1357)
- Llama:  slot 8 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent history/d3faab74_11218.txt (rel 0), challenger history/0330a753_207102.txt (rel 0), true delta +0.000

INCUMBENT: At 1055, finally responding to Admiral Nimitz's message, Admiral Halsey ordered USS New Jersey to lead Task Force 34, and Admiral Bogan’s carrier group, to rescue Taffy 3. At 1115 when, as Halsey later stated, the remnants of Admiral Ozawa’s northern force “was exactly 42 miles from the muzzles of my 16‑inch guns” on the New Jersey, she, five other battleships, and the rest of Task Force 34 reversed course and headed south. By that time, Task Force 34 was so far from San Bernardino Strait that it could not get there for more than thirteen hours given the slower speed of its older battleships. 

CHALLENGER: Since then the story has been different; for the naval victory of Leyte Gulf totally changed the relationship of naval forces, Jap and American. When General MacArthur's divisions landed on the beaches of Lingayan Gulf last December to grasp Luzon, they had to fight only the garrison of that island. There were no reinforcements this time, and no stealthy shuttling in of supplies for the Jap forces, because we fully controlled the adjacent seas. The proof of our command of those seas was plain, for the Fast Carrier Task Force, after vicious stabs at the Jap airfields of Formosa and Luzon, steam

Pair slot 9: incumbent history/8005e445_11217.txt (rel 0), challenger history/0330a753_207102.txt (rel 0), true delta +0.000

INCUMBENT: By contrast, Admiral Kurita’s more powerful force debouched from San Bernardino Strait at 0035, expecting a fight but finding no ships guarding the strait. The Japanese decoy plan had succeeded. The main Japanese force sailed unchallenged around the island of Samar toward the transports in Leyte Gulf while the two American battle fleets were far to the north and south. The only thing between Admiral Kurita's fleet and the American amphibious ships were the Seventh Fleet’s three groups of escort carriers. Directly before the Japanese onslaught was Task Group 77.4.3, with the radio call signal “

CHALLENGER: Since then the story has been different; for the naval victory of Leyte Gulf totally changed the relationship of naval forces, Jap and American. When General MacArthur's divisions landed on the beaches of Lingayan Gulf last December to grasp Luzon, they had to fight only the garrison of that island. There were no reinforcements this time, and no stealthy shuttling in of supplies for the Jap forces, because we fully controlled the adjacent seas. The proof of our command of those seas was plain, for the Fast Carrier Task Force, after vicious stabs at the Jap airfields of Formosa and Luzon, steam

## 10. history / 69226_361 — gap 0.1357
Query: Why did water stored on premodern ships &quot;go brackish&quot; or &quot;go bad&quot; while at sea?
<p>I have read in several Napoleonic historical fiction novels that water stored on ships would somehow spoil over time. The water is described as completely unpalatable if not undrinkable, and alcohol (rum, beer) is often added as a preservative. I have found many sources online describing the act of lightly boozing up the water supply to keep it from &quot;going bad&quot;, but I am having trouble finding information on <strong>what exactly happens to the water</strong>. One of my historical fiction books described the water &quot;going brackish&quot;, but I can't imagine how salt would get in there. Is it just bacterial growth? Tannins from the wood?</p>

- Direct: none (query delta 0.0)
- Llama:  slot 10 delta -0.136 (query delta -0.1357)

Pair slot 8: incumbent history/b2c2b54f_239317.txt (rel 0), challenger history/d5a0d3fc_239294.txt (rel 0), true delta +0.000

INCUMBENT: How Do You Desalinate Brackish Water? When confronted with brackish conditions water characterized by a salinity level higher than freshwater but lower than seawater the efficacy of reverse osmosis desalination systems becomes particularly evident. Reverse osmosis stands out as an ideal option for purifying brackish water due to its energy efficiency, versatility, and cost-effectiveness. By leveraging high-pressure filtration, our desalination systems effectively address the unique challenges posed by brackish water sources, providing a reliable and sustainable solution for communities and ind

CHALLENGER: Brackish water is treated with a water treatment system that is rated for desalination. Desalination is a process that removes dissolved mineral salts from saline water and converts it into fresh, drinkable water. Reverse osmosis and distillation are the primary desalination technologies used to treat brackish water. Seawater can also be desalinated through reverse osmosis and distillation, but the desalination of sweater is not as efficient due to higher salt content.

Pair slot 9: incumbent history/775b9ef6_334766.txt (rel 0), challenger history/d5a0d3fc_239294.txt (rel 0), true delta +0.000

INCUMBENT: Boteler 1634:  when it stinketh much, it is a sign that the water hath lain long in the hold of the ship; and on the contrary, when it is clear and sweet, it is a token that it comes freshly in from the sea. This stinking water therefore is always a welcome perfume to an old seaman; and he that stops his nose at it is laughed at, and held but a fresh-water man at best.

CHALLENGER: Brackish water is treated with a water treatment system that is rated for desalination. Desalination is a process that removes dissolved mineral salts from saline water and converts it into fresh, drinkable water. Reverse osmosis and distillation are the primary desalination technologies used to treat brackish water. Seawater can also be desalinated through reverse osmosis and distillation, but the desalination of sweater is not as efficient due to higher salt content.

## 11. history / 72726_139 — gap 0.1357
Query: Is there a dog in Rembrandt&#39;s &quot;The Night Watch&quot;?
<p>This, of course, is Rembrandt's <em>The Night Watch</em></p>
<p><img src="https://cdn.britannica.com/98/182298-050-ECE18F92/Frans-Banninck-Cocq-Militia-Company-of-District.jpg" alt="The Night Watch" /></p>
<p>In the bottom right, I swear there's a figure of a dog or something, but I can't see anything about there being a dog in the painting.</p>
<p>Does anybody know what it actually is?</p>

- Direct: slot 10 delta +0.136 (query delta 0.1357)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/aab030bc_18793.txt (rel 0), challenger history/d71f4df7_18807.txt (rel 0), true delta +0.000

INCUMBENT: It's a dog. Not only a good scan like that in https://www.rijksmuseum.nl/en/collection/SK-C-5 shows it is a dog, but sources like https://www.smithsonianmag.com/smart-news/curators-discover-hidden-sketch-in-rembrandts-night-watch-180979196/ mention it: "New research also indicates that a dog at the bottom of the composition has faded over the centuries". 1 A Netherlands-based history account I follow was just saying earlier this week how Dutch "masters" paintings tend to contain background dogs. T.E.D. – T.E.D. ♦ 2023-11-03 17:26:14 +00:00 Commented Nov 3, 2023 at 17:26 Start asking to get ans

CHALLENGER: An oil on panel painting depicitng an interior guard room scene with soldiers and a dog. The central figure in the painting, an officer in uniform stands in command, looking at the soldiers on his left and pointing to what appears to be some broken sticks on the floor to his right. He seems to be accusing the solders of cluttering his floor with broken sticks, as the the dog next to him is likely the real culprit. The soldier is wearing his gorget and reaching for his cuirass. On the far right, a red standard and drum. There is a rapier sword on the floor. In the background are many other sold

Pair slot 9: incumbent history/180da4ee_182903.txt (rel 0), challenger history/d71f4df7_18807.txt (rel 0), true delta +0.000

INCUMBENT: When I saw the drawing in an exhibition,  says Anne Lenders , curator of Operation Night Watch ,  I immediately thought of the dog in Night Watch . The dog's head, collar, and pose bear such a strong resemblance that one can only assume that Rembrandt used this drawing as inspiration. Subsequent research confirmed this.

CHALLENGER: An oil on panel painting depicitng an interior guard room scene with soldiers and a dog. The central figure in the painting, an officer in uniform stands in command, looking at the soldiers on his left and pointing to what appears to be some broken sticks on the floor to his right. He seems to be accusing the solders of cluttering his floor with broken sticks, as the the dog next to him is likely the real culprit. The soldier is wearing his gorget and reaching for his cuirass. On the far right, a red standard and drum. There is a rapier sword on the floor. In the background are many other sold

## 12. politics / 87499_976 — gap 0.1357
Query: Did Marx believe that politics is left unchanged by Historical Materialism?
<p>Recently, I've been reading the works of Marx and Engels to understand the meaning of Historical Materialism; as I inclusively believe in a form of Collective Determinism of ideas ( not physical or revolutionary actions, nor individual ) enforced in all hitherto of the society to certain classes, although I do not certainly acknowledge if I can objectify my beliefs as Historical Materialism.</p>
<p>While reading the chapter two of the <em>Communist Manifesto</em>, I faced this phrase:</p>
<blockquote>
<p><strong>&quot;But, &quot; it will be said, &quot;Religious, Moral, Philosophical, juridical ideas, etc. have indeed been modified in the course of historical development.
Religion, Morality, Philosophy, Politics, Law, constantly survived this exchange&quot;</strong></p>
</blockquote>
<p>My point of interest is the second segment of this saying: <br></p>
<ol>
<li>What does the author mean by this?</li>
<li>Does the author want to indicate that politics are not affected by Historical Materialism, or that they are themselves the cause of it?</li>
</ol>

- Direct: slot 10 delta +0.136 (query delta 0.1357)
- Llama:  slot 10 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent politics/04c1f986_15271.txt (rel 0), challenger politics/4854c5fb_15162.txt (rel 1), true delta +0.000

INCUMBENT: Karl Marx (1818–1883) was a German philosopher and economist. In 1848 he and Friedrich Engels (1820–1895) co-authored the Communist Manifesto. This book is one of the most influential political manuscripts in history. It also presents in a highly condensed form Marx’s theory of society, which differed from what Comte proposed. Whereas Comte viewed the goal of sociology as recreating a unified, post-feudal spiritual order that would help to institutionalize a new era of political and social stability, Marx developed a critical analysis of capitalism that saw the material or economic basis of in

CHALLENGER: At its core, historical materialism is a way to understand history through the changes in economic systems. Marx argued that the economic base of a society shapes its superstructure, which includes politics, culture, and ideology. Simply put, the way we produce things and the relationships that arise from this production form the foundation upon which everything else is built.

The economic base 🔗
The economic base consists of two main components:

Mode of production: This refers to the way goods and services are produced. For instance, in a capitalist society, production is driven by private 

Pair slot 9: incumbent politics/87499_8b27f360bca000d6_0.txt (rel 0), challenger politics/4854c5fb_15162.txt (rel 1), true delta +0.000

INCUMBENT: The real originality of Marx and Engels lies in the field of politics, not in economics or philosophy. They were the first to discover the historical potential of the new class that capitalism had brought into existence—the modern proletariat, a class that could encompass a universal liberation from all prevailing forms of oppression and exploitation. The modern workers’ movement, capable of self-determination and self-emancipation, able to draw on the best of bourgeois culture and science, would have no need of utopias or religious exaltation. The political capacity of the proletariat sprang 

CHALLENGER: At its core, historical materialism is a way to understand history through the changes in economic systems. Marx argued that the economic base of a society shapes its superstructure, which includes politics, culture, and ideology. Simply put, the way we produce things and the relationships that arise from this production form the foundation upon which everything else is built.

The economic base 🔗
The economic base consists of two main components:

Mode of production: This refers to the way goods and services are produced. For instance, in a capitalist society, production is driven by private 

## 13. history / 56068_771 — gap 0.1232
Query: Why did the Nazis collaborate with the Ustaše?
<p>The Croats are linguistically, ethnically and culturally a Slavic people. While the Ustaše did not see themselves as such, why did the Nazis choose to collaborate with what was essentially a nationalist organisation for a Slavic ethnic group?</p>

- Direct: slot 8 delta +0.123 (query delta 0.1232)
- Llama:  slot 10 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent history/9af06ac5_349578.txt (rel 0), challenger history/25a88ea4_349707.txt (rel 0), true delta +0.000

INCUMBENT: The ideology of the movement was a blend of fascism, Roman Catholicism and Croatian ultranationalism. The Ustaše supported the creation of a Greater Croatia that would span the Drina River and extend to the border of Belgrade. The movement advocated a racially  pure  Croatia and promoted genocide against Serbs due to the Ustaše's anti-Serb sentiment and Holocaust against Jews and Roma via Nazi racial theory, and persecution of anti-fascist or dissident Croats and Bosniaks. The Ustaše viewed the Bosniaks as  Muslim Croats , and as a result, Bosniaks were not persecuted on the basis of race. The

CHALLENGER: The ideology of the movement was a blend of fascism , Roman Catholicism and Croatian ultranationalism .   44   The Ustaše supported the creation of a Greater Croatia that would span the Drina River and extend to the border of Belgrade .   47   The movement advocated a racially  pure  Croatia and promoted genocide against Serbs-due to the Ustaše's anti-Serb sentiment -and Holocaust against Jews and Roma via Nazi racial theory , and persecution of anti-fascist or dissident Croats and Bosniaks. The Ustaše viewed the Bosniaks as   Muslim Croats  , and as a result, Bosniaks were not persecuted on t

Pair slot 9: incumbent history/b9feb9cc_349720.txt (rel 0), challenger history/25a88ea4_349707.txt (rel 0), true delta +0.000

INCUMBENT: Nazi Germany initially did not support an independent Croatia, nor did it support the Ustaše, with Hitler stressing the importance of a  strong and united Yugoslavia .   71   Nazi officials, including Hermann Göring , wanted Yugoslavia stable and officially neutral during the war so Germany could continue to securely gain Yugoslavia's raw material exports.   71   The Nazis grew irritated with the Ustaše, among them Reichsfuhrer SS Heinrich Himmler , who was dissatisfied with the lack of full compliance by the NDH to the Nazis' agenda of extermination of the Jews, as the Ustaše permitted Jews w

CHALLENGER: The ideology of the movement was a blend of fascism , Roman Catholicism and Croatian ultranationalism .   44   The Ustaše supported the creation of a Greater Croatia that would span the Drina River and extend to the border of Belgrade .   47   The movement advocated a racially  pure  Croatia and promoted genocide against Serbs-due to the Ustaše's anti-Serb sentiment -and Holocaust against Jews and Roma via Nazi racial theory , and persecution of anti-fascist or dissident Croats and Bosniaks. The Ustaše viewed the Bosniaks as   Muslim Croats  , and as a result, Bosniaks were not persecuted on t

## 14. history / 56254_754 — gap 0.1232
Query: Was Yahweh a god in a possible Edomite Pantheon?
<p>I understand from the Bible, from the Wikipedia articles <a href="https://en.wikipedia.org/wiki/Yahweh" rel="nofollow noreferrer">Yahweh</a> and <a href="https://en.wikipedia.org/wiki/Yahwism" rel="nofollow noreferrer">Yahwism</a> and from lectures of various scholars that polytheism was practiced by at least some religious leaders, priests, kings and laymen of the allegedly-Canaanite kingdom of Israel and the allegedly-Canaanite Kingdom of Judah:</p>

<p>One opinion is that Yahweh was a war or storm god worshiped by at least one (probably Midianite, probably aniconic) group, and that this group later migrated northern and absorbed into ancient Israeli-Judean society and introduced Yahweh whom was adopted as the national god of the two kingdoms but not necessarily in an aniconic way and often presented with Asherah as his consort.</p>

<p><a href="https://he.wikipedia.org/wiki/%D7%99%D7%92%D7%90%D7%9C_%D7%91%D7%9F-%D7%A0%D7%95%D7%9F" rel="nofollow noreferrer">Yigal Ben Nun</a> who wrote the book "<a href="https://he.wikipedia.org/wiki/%D7%A7%D7%99%D7%A6%D7%95%D7%A8_%D7%AA%D7%95%D7%9C%D7%93%D7%95%D7%AA_%D7%99%D7%94%D7%95%D7%94" rel="nofollow noreferrer">קיצור תולדות יהוה</a>" (no English name yet) often claims that the first Israelite temple included a statue of Yahweh and possibly of Asherah and that probably was a statue of any other god in its temple ("house") per wider Canaanite pantheon and that there are numerous clues for that in the Hebrew bible itself;<br>
Besides that, some allege that a monotheistic <a href="https://en.wikipedia.org/wiki/Aniconism" rel="nofollow noreferrer">aninconic</a> "yahwistic" sect was slowly developing primarily in Judean society, and adopted by aristocracy gradually to remove elements of polytheism until finally consolidated as the sole form of Israelite religion in <a href="https://en.wikipedia.org/wiki/Babylonian_captivity" rel="nofollow noreferrer">Babylonian captivity</a>.</p>

<p>Many archaeologists and Bible researchers opinionated about a merging of Yahweh, El and Asherah into Yahweh itself in Israel and Judah by that sect:</p>

<blockquote>
  <ul>
  <li><strong><a href="https://en.wikipedia.org/wiki/El_(deity)" rel="nofollow noreferrer">El</a></strong> (a named <code>[Canaanite][supreme]</code> god) →→→ el ("god" <code>[single]</code>) or eloah ("god" <code>[single]</code>) or elohim ("god" <code>[plural]</code>)</li>
  <li><strong>el's/eloah's/elohim's name</strong> →→→ Yahweh</li>
  <li><p><strong><a href="https://en.wikipedia.org/wiki/Asherah" rel="nofollow noreferrer">Asherah</a></strong> (El's consort) →→→ Yahweh's consort and later <a href="https://en.wikipedia.org/wiki/Shekhinah" rel="nofollow noreferrer">Shekhinah</a> ("presence" or "effect" of god) and/or <a href="https://en.wikipedia.org/wiki/Menorah_(Temple)" rel="nofollow noreferrer">Menorah</a> (as a religious item) which are feminine terms in Hebrew, adjusted to the fertility of Asherah as a fertility goddess or as being symbolized by a tree fertile with branches ("Menorah branches"), respectively<br>
  <sub>These Asherah ideas might be supported from findings at <a href="https://en.wikipedia.org/wiki/Kuntillet_Ajrud" rel="nofollow noreferrer">Kuntillet Ajrud</a>.</sub></p></li>
  <li><p><strong>Ashdat</strong> →→→ X</p></li>
  </ul>
</blockquote>

<p>The bible is allegedly cluing that Yahweh first "appeared" or worshiped in areas named:<br>
<em>Sinai</em>, <em>Horeb</em> <em>Teman</em>, <em>Paran</em>, <em>Seir</em> and <em>Edom</em></p>

<p>These names might describe an area stretching primarily from western Sinai peninsula to the mountains of Edom, while:</p>

<ul>
<li>Teman might be a general term for that southern area (including further south until <a href="https://en.wikipedia.org/wiki/Tayma" rel="nofollow noreferrer">Tayma</a>)</li>
<li>Paran might be the slopes along Edom mountain(s)</li>
<li>Seir as the mountain (or mountain range) named Edom mountain(s)</li>
<li>Edom as an inclusion of Paran and Seir</li>
<li>Horeb might be a specific area within Sinai</li>
</ul>

<hr>

<p><strong>Deuteronomy 33:2 (KJV)</strong><br>
<strong>דברים פרק לג פסוק ב</strong></p>

<blockquote>
  <p>וַיֹּאמַ֗ר יְהוָ֞ה מִסִּינַ֥י בָּא֙ וְזָרַ֤ח מִשֵּׂעִיר֙ לָ֔מוֹ הוֹפִ֙יעַ֙
  מֵהַ֣ר פָּארָ֔ן וְאָתָ֖ה מֵרִבְבֹ֣ת קֹ֑דֶשׁ מִימִינ֕וֹ אֵ֥שׁדָּ֖ת אֵ֥שׁ דָּ֖ת
  לָֽמוֹ</p>
</blockquote>

<p>Which I would translate as:</p>

<p><em>And he said: Yahweh came from Sinai and shined from Seir for them ; (he) appeared from Paran mountain(s?) ; and with him from(among?) tens of thousands of holy ones ; Ashdat from his right ; Ash Dat (?) for them.</em><br>
<sub><strong>Note:</strong> Ashdat part isn't included in KJV <a href="https://biblehub.com/text/deuteronomy/33-2.htm" rel="nofollow noreferrer">but is included here</a>.</sub></p>

<hr>

<p><strong>Judges 5:4 (KJV)<br></strong>
<strong>שופטים פרק ה פסוק ד</strong></p>

<blockquote>
  <p>יְהוָה בְּצֵאתְךָ מִשֵּׂעִיר בְּצַעְדְּךָ מִשְּׂדֵה אֱדוֹם אֶרֶץ
  רָעָשָׁה גַּם שָׁמַיִם נָטָפוּ גַּם עָבִים נָטְפוּ מָיִם.</p>
</blockquote>

<p>Which I would translate as:</p>

<p><em>Yahweh, in your exit from Seir, in your marching from Edom field ; (the) land was noised, also skies fell, also clouds felled water.</em></p>

<hr>

<p><strong>Habakkuk 3:3 (KJV)</strong><br>
<strong>חבקוק פרק ג פסוק ג</strong></p>

<blockquote>
  <p>אֱלוֹהַ מִתֵּימָן יָבוֹא וְקָדוֹשׁ מֵהַר פָּארָן סֶלָה כִּסָּה
  שָׁמַיִם הוֹדוֹ וּתְהִלָּתוֹ מָלְאָה הָאָרֶץ.</p>
</blockquote>

<p>Which I would translate as:</p>

<p><em>Eloah from Teman would come and a holy one from Paran mountain(s?) ; <a href="https://en.wikipedia.org/wiki/Selah" rel="nofollow noreferrer">selāh</a> ; his beauty covered the sky and his glory (was) filled (in) the land.</em></p>

<hr>

<h2>My problem</h2>

<p>All of this might clue that Yahweh was not only worshiped in a pantheon of at least some Ancient Israelis and at least some ancient Judeans, but maybe also by ancient Edomites ; Edomites had the national god <a href="https://en.wikipedia.org/wiki/Qos_(deity)" rel="nofollow noreferrer">Qos</a> and maybe were the closest kingdom in culture and language to Israel and Judea → some researchers <a href="https://en.wikipedia.org/wiki/Edomite_language#cite_note-Glottolog2-2" rel="nofollow noreferrer">consider Edomite language as a dialect of Hebrew</a>.</p>

<h2>My question</h2>

<p>Was Yahweh a god in a possible Edomite Pantheon?</p>

- Direct: slot 8 delta +0.123 (query delta 0.1232)
- Llama:  slot 8 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent history/1a6f95ff_346621.txt (rel 0), challenger history/70ace2c5_346576.txt (rel 0), true delta +0.000

INCUMBENT: In the national crisis of the Babylonian exile, Yahweh is described as the sole deity and absorbs all attributes of previous gods and goddesses. The notion of Yahweh as a supreme deity is described in the 6th-century BCE Second Isaiah. The author's praise for Yahweh is motivated by restoring Israel's confidence into their own historical gods against the deities of their Babylonian enemies. The claim for monotheism is directed against the deities of Nebuchadnezzar II, who founded his reign on Marduk and Nabu. The transition was a gradual one and was not totally accomplished during the First Tem

CHALLENGER: From the perspective of the Kenite hypothesis, it has also been suggested that the Edomite deity Qōs might have been one and the same as Yahweh, rather than a separate deity, with its name a title of the latter. Aside from their common territorial origins, various common characteristics between the Yahwist cult and the Edomite cult of Qōs hint at a shared connection. Doeg the Edomite, for example, is depicted as having no problem in worshiping Yahweh and is shown to be at home in Jewish sanctuaries.

Pair slot 9: incumbent history/bd2afc44_137515.txt (rel 0), challenger history/70ace2c5_346576.txt (rel 0), true delta +0.000

INCUMBENT: Yahweh, the national god of both Israel and Judah, seems to have originated in Edom and Midian in southern Canaan and may have been brought to Israel by the Kenites and Midianites at an early stage.   180

CHALLENGER: From the perspective of the Kenite hypothesis, it has also been suggested that the Edomite deity Qōs might have been one and the same as Yahweh, rather than a separate deity, with its name a title of the latter. Aside from their common territorial origins, various common characteristics between the Yahwist cult and the Edomite cult of Qōs hint at a shared connection. Doeg the Edomite, for example, is depicted as having no problem in worshiping Yahweh and is shown to be at home in Jewish sanctuaries.

## 15. history / 56861_715 — gap 0.1232
Query: How was the first carpenter&#39;s square made?
<p>A "square" is a tool, typically L or triangle shaped, that gives a craftsperson a quick and accurate 90° angle to use as a baseline in many important trades.  </p>

<p>It's easy to find a right angle if you already have a square, but how was the first square made?  How can someone create a right angle tool "from scratch"?</p>

- Direct: slot 8 delta -0.123 (query delta -0.1232)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/847ee096_101352.txt (rel 1), challenger history/fa666fb1_341636.txt (rel 0), true delta +0.000

INCUMBENT: In early nineteenth-century references to squares, the eighteen-inch by twelve-inch size is usual, but the tool has been around for centuries. It was used in Europe long before it was first patented in this country in 1819 by one Silas Hawes, a blacksmith in South Shaftsbury, Vermont.



CHALLENGER: note to Jay:When I began to research steel rules in 2008, the reports I could find believed that Hawes invented the square in Shaftsbury, VT. I had a very hard time finding out about simple things, like tools in use in the United States before 1800. So I have been recording what I can find, as in this post. I am writing another post with more information. Thanks for adding your knowledge here. I am reading Will's bog as well. Very interesting -keep it up, please.

Pair slot 9: incumbent history/c58a734e_101396.txt (rel 0), challenger history/fa666fb1_341636.txt (rel 0), true delta +0.000

INCUMBENT: You can then create an L-shaped "square" by eye and check it against your straight edge. The method is to lay one arm of your square against the straight edge of a plank and mark a first perpendicular line across the plank along the other edge of the square, you then flip the square over, align the square where the line meets the straight-edge and draw a second perpendicular line across the plank, this will likely form a narrow V because your square is not exactly right. You can see how far out your square is and make appropriate adjustments, repeating the process until you are no longer able 

CHALLENGER: note to Jay:When I began to research steel rules in 2008, the reports I could find believed that Hawes invented the square in Shaftsbury, VT. I had a very hard time finding out about simple things, like tools in use in the United States before 1800. So I have been recording what I can find, as in this post. I am writing another post with more information. Thanks for adding your knowledge here. I am reading Will's bog as well. Very interesting -keep it up, please.

## 16. history / 76952_71 — gap 0.1232
Query: Why did German typewriters not use Fraktur?
<p><a href="https://en.wikipedia.org/wiki/Fraktur" rel="noreferrer">Fraktur</a> was the dominant typeface in Germany for centuries until the Nazis suddenly stopped using it in WWII.</p>
<p>I read <sup>citation needed</sup> that the Nazis, before they suddenly marked Fraktur as &quot;Jewish&quot; in 1941, tried to introduce Fraktur typewriters in 1933 but quickly gave up.</p>
<p>That inspired my questions: Why were German typewriters not using Fraktur in the first place?</p>
<p>The Nazis stopped using Fraktur in 1941, officially because they marked it &quot;Jewish&quot;, but probably because they found it impractical for their &quot;empire&quot; with many non-German speaking regions. But typewriters are much older than that. I wonder, if someone build a typewriter in Germany in 1920, why did they use Antiqua (latin letters) instead of Fraktur?</p>

- Direct: slot 8 delta -0.123 (query delta -0.1232)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/43e4284a_9573.txt (rel 1), challenger history/f21a6fcf_144669.txt (rel 0), true delta +0.000

INCUMBENT: Ironically, Fraktur’s downfall came not from foreign influence but from within. In 1941, the Nazi regime abruptly banned Fraktur, labeling it “Schwabacher Judenlettern” (Schwabacher Jewish letters) — a wildly inaccurate claim likely rooted in convenience and propaganda. The shift aimed to align with international modernism and make German writing more legible abroad. This sudden rejection, after centuries of use, left Fraktur tainted by association and hastened its decline in everyday life.


Fraktur Today: Revival and Reclamation
In recent decades, Fraktur has experienced a cautious but creat

CHALLENGER: From the late 18th century to the late 19th century, Fraktur was progressively replaced by Antiqua) as a symbol of the classicist age and emerging cosmopolitanism in most of the countries in Europe that had previously used Fraktur. This move was hotly debated in Germany, a controversy known as the Antiqua-Fraktur dispute . The shift affected mostly scientific writing in Germany, whereas most belletristic literature and newspapers continued to be printed in Fraktur.

Pair slot 9: incumbent history/476fca56_133076.txt (rel 0), challenger history/f21a6fcf_144669.txt (rel 0), true delta +0.000

INCUMBENT: Blackletter was used throughout Western Europe from approximately 1150 until the 17th century. It continued to be commonly used for Danish, Norwegian, and Swedish until the 1870s, Finnish until the turn of the 20th century, Estonian and Latvian until the 1930s, and for the German language until the 1940s, when Adolf Hitler officially banned it in 1941. Fraktur is a notable script of this type, and sometimes the entire group of blackletter faces is referred to as Fraktur. Blackletter is not to be confused with the Old English language, which predates blackletter by many centuries and was writte

CHALLENGER: From the late 18th century to the late 19th century, Fraktur was progressively replaced by Antiqua) as a symbol of the classicist age and emerging cosmopolitanism in most of the countries in Europe that had previously used Fraktur. This move was hotly debated in Germany, a controversy known as the Antiqua-Fraktur dispute . The shift affected mostly scientific writing in Germany, whereas most belletristic literature and newspapers continued to be printed in Fraktur.

## 17. quant / 69707_1089 — gap 0.1232
Query: Implied volatilities for different options that track the same stock
<p>I have a somewhat basic question regarding option prices.</p>
<p>Suppose we have an underlying stock and two different options (that have different strike prices, maturities, etc.) that track this stock. What factors can cause these options to have different implied volatilities?</p>
<p>One intuitive reason is I've come up with is maturity<span class="math-container">$-$</span>a longer maturity option should price in longer-term volatility than the shorter maturity option.</p>
<p>What are some other factors?</p>

- Direct: slot 8 delta +0.123 (query delta 0.1232)
- Llama:  slot 10 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent quant/8c98e929_28771.txt (rel 0), challenger quant/69707_ca45b9d7f2ca5c2e_2.txt (rel 0), true delta +0.000

INCUMBENT: The implied volatility in the formula is the volatility of the underlying instrument. Some traders believe the implied volatility of an option is a more useful measure of an option's relative value than the price. Traders should also consider the implied volatility of an option on a dividend-paying stock. The higher the implied volatility of a stock, the more likely the price will go down. Thus, the implied volatility on put options is higher leading up to the ex-dividend date due to the price drop.

CHALLENGER: Implied volatility (IV) is calculated as the value that fits an option model’s price to the market price. IV is interpreted as the expected volatility implied by the market, while historical volatility reflects the past trading range of an underlying asset. Retail option traders may be more interested in IV due to its forward-looking nature. It estimates an underlying asset’s future price movement according to changes in the price of a corresponding option and can be used as a useful approach in comparing prices for different underlying assets, strike prices, and maturities. Traders may quote 

Pair slot 9: incumbent quant/78042_fa82b31999d8bb06_616.txt (rel 0), challenger quant/69707_ca45b9d7f2ca5c2e_2.txt (rel 0), true delta +0.000

INCUMBENT: 15.8.3 Breakeven Share Price and Volatility In order to assess the option strategy, investors often prepare a proﬁt/loss chart. This chart can be used to evaluate the proﬁt/loss of the strategy at different  prices. Moreover, option pricing theory can be applied to better understand the breakeven  price points in the chart. The option price depends on the expected price volatility of the underlying. The higher (lower) the volatility, the higher (lower) the option price. The implied volatility of an option can be calculated with the traded option price and an option pricing model such as the Bl

CHALLENGER: Implied volatility (IV) is calculated as the value that fits an option model’s price to the market price. IV is interpreted as the expected volatility implied by the market, while historical volatility reflects the past trading range of an underlying asset. Retail option traders may be more interested in IV due to its forward-looking nature. It estimates an underlying asset’s future price movement according to changes in the price of a corresponding option and can be used as a useful approach in comparing prices for different underlying assets, strike prices, and maturities. Traders may quote 

## 18. hsm / 5627_754 — gap 0.1175
Query: When was it discovered that the Earth wasn&#39;t round?
<p>We know the Earth isn't a sphere: that is, the equatorial circumference isn't equal to the polar circumference.  When (and how) was this discovered?</p>

<p>I can put a lower bound of around the 3rd century BC, when Greek astronomy showed that the Earth was round rather than flat, and an upper bound of 1791, when the French Academy of Sciences specified a quadrant of <em>meridian</em> in the definition of the meter.  An uncertainty of 2000 years is rather large, though.</p>

- Direct: slot 9 delta +0.118 (query delta 0.1175)
- Llama:  slot 9 delta +0.000 (query delta 0.0)

Pair slot 8: incumbent hsm/13022_b14c28c921047a97_2.txt (rel 0), challenger hsm/5627_fc1fc4b4ceb8955b_349.txt (rel 0), true delta +0.000

INCUMBENT: Early in Classical Greece, knowledge that the Earth is spherical ("round") was common. Around 240 BCE, as the result of a seminal experiment, Eratosthenes (276–194 BCE) accurately estimated its circumference. In contrast to Aristotle's geocentric views, Aristarchus of Samos (Greek: Ἀρίσταρχος; c. 310 – c. 230 BCE) presented an explicit argument for a heliocentric model of the Solar System, i.e. for placing the Sun, not the Earth, at its centre. Seleucus of Seleucia, a follower of Aristarchus' heliocentric theory, stated that the Earth rotated around its own axis, which, in turn, revolved aroun

CHALLENGER: **Earth**, the planet which we inhabit, a nearly spherical body which every twenty-four hours rotates from west to east round an imaginary line called its axis—this axis having as its extremities the north and south poles—while in the course of a year it completes a revolution round the sun. To an observer whose view is not obstructed, the visible part of the earth appears as a circular and horizontal expanse, on the circumference of which the heavens appear to rest. Accordingly, in remote antiquity, the earth was regarded as a flat, circular body, floating on the waters. But even in antiquity

Pair slot 9: incumbent hsm/17737_5cbc4d797ed05070_486.txt (rel 0), challenger hsm/5627_fc1fc4b4ceb8955b_349.txt (rel 0), true delta +0.000

INCUMBENT: Operations for determining the figure of the earth were carried out during the last century on an unprecedented scale. The Russo-Scandinavian arc, of which the measurement was completed under the direction of the elder Struve in 1855, reached from Hammerfest to Ismailia on the Danube, a length of 25° 20′. But little inferior to it was the Indian arc, begun by Lambton in the first years of the century, continued by Everest, revised and extended by Walker. Both were surpassed in compass by the Anglo-French arc, which embraced 28°; and considerable segments of meridians near the Atlantic and Paci

CHALLENGER: **Earth**, the planet which we inhabit, a nearly spherical body which every twenty-four hours rotates from west to east round an imaginary line called its axis—this axis having as its extremities the north and south poles—while in the course of a year it completes a revolution round the sun. To an observer whose view is not obstructed, the visible part of the earth appears as a circular and horizontal expanse, on the circumference of which the heavens appear to rest. Accordingly, in remote antiquity, the earth was regarded as a flat, circular body, floating on the waters. But even in antiquity

## 19. history / 70695_262 — gap 0.1128
Query: Was there really a whale skeleton in Jaffa?
<p>Was there really a whale skeleton in <a href="https://en.wikipedia.org/wiki/Jaffa" rel="noreferrer">Jaffa</a> or at least a common myth about it?</p>
<p>I cannot find any historical reference to it but still believe, from the way he wrote Moby Dick, that Melville did not make up this fact:</p>
<blockquote>
<p>for in the ancient Joppa, now Jaffa, on the Syrian coast, in one of the Pagan temples, there stood for many ages the vast skeleton of a whale, which the city's legends and all the inhabitants asserted to be the identical bones of the monster that Perseus slew. When the Romans took Joppa, the same skeleton was carried to Italy in triumph. What seems most singular and suggestively important in this story, is this: it was from Joppa that Jonah set sail.</p>
<p>Source: <a href="http://www.powermobydick.com/Moby082.html" rel="noreferrer"><em>Moby Dick</em> p. 360</a></p>
</blockquote>

- Direct: slot 10 delta +0.113 (query delta 0.1128)
- Llama:  none (query delta 0.0)

Pair slot 8: incumbent history/ea6fbeff_36180.txt (rel 0), challenger history/0eaafd96_36012.txt (rel 0), true delta +0.000

INCUMBENT: The intense Pequod sailed on; the rolling waves and days went by; the life-buoy-coffin still lightly swung; and another ship, most miserably misnamed the Delight, was descried. As she drew nigh, all eyes were fixed upon her broad beams, called shears, which, in some whaling-ships, cross the quarter-deck at the height of eight or nine feet; serving to carry the spare, unrigged, or disabled boats. Upon the stranger’s shears were beheld the shattered, white ribs, and some few splintered planks, of what had once been a whale-boat; but you now saw through this wreck, as plainly as you see through t

CHALLENGER: EXTRACTS (Supplied by a Sub-Sub-Librarian). CHAPTER 1. Loomings. CHAPTER 2. The Carpet-Bag. CHAPTER 3. The Spouter-Inn. CHAPTER 4. The Counterpane. CHAPTER 5. Breakfast. CHAPTER 6. The Street. CHAPTER 7. The Chapel. CHAPTER 8. The Pulpit. CHAPTER 9. The Sermon. CHAPTER 10. A Bosom Friend. CHAPTER 11. Nightgown. CHAPTER 12. Biographical. CHAPTER 13. Wheelbarrow. CHAPTER 14. Nantucket. CHAPTER 15. Chowder. CHAPTER 16. The Ship. CHAPTER 17. The Ramadan. CHAPTER 18. His Mark. CHAPTER 19. The Prophet. CHAPTER 20. All Astir. CHAPTER 21. Going Aboard. CHAPTER 22. Merry Christmas. CHAPTER 23. The Lee 

Pair slot 9: incumbent history/268d7602_36152.txt (rel 0), challenger history/0eaafd96_36012.txt (rel 0), true delta +0.000

INCUMBENT: In the first place, I wish to lay before you a particular, plain statement, touching the living bulk of this leviathan, whose skeleton we are briefly to exhibit. Such a statement may prove useful here. According to a careful calculation I have made, and which I partly base upon Captain Scoresby’s estimate, of seventy tons for the largest sized Greenland whale of sixty feet in length; according to my careful calculation, I say, a Sperm Whale of the largest magnitude, between eighty-five and ninety feet in length, and something less than forty feet in its fullest circumference, such a whale will

CHALLENGER: EXTRACTS (Supplied by a Sub-Sub-Librarian). CHAPTER 1. Loomings. CHAPTER 2. The Carpet-Bag. CHAPTER 3. The Spouter-Inn. CHAPTER 4. The Counterpane. CHAPTER 5. Breakfast. CHAPTER 6. The Street. CHAPTER 7. The Chapel. CHAPTER 8. The Pulpit. CHAPTER 9. The Sermon. CHAPTER 10. A Bosom Friend. CHAPTER 11. Nightgown. CHAPTER 12. Biographical. CHAPTER 13. Wheelbarrow. CHAPTER 14. Nantucket. CHAPTER 15. Chowder. CHAPTER 16. The Ship. CHAPTER 17. The Ramadan. CHAPTER 18. His Mark. CHAPTER 19. The Prophet. CHAPTER 20. All Astir. CHAPTER 21. Going Aboard. CHAPTER 22. Merry Christmas. CHAPTER 23. The Lee 

## 20. law / 101678_842 — gap 0.1128
Query: Public Access to Records of Criminal Proceedings
<p>Follow-up to <a href="https://law.stackexchange.com/questions/101675/victim-anonymity-in-criminal-proceedings">this question</a></p>
<p>Alice is assaulted by Bob.  Alice reports the assault to the police,
including the fact that Bob is the assailant. The police investigate
then turn their findings over to the local prosecutor's office.  After
reviewing the case, the prosecutor's office brings charges and issues an
arrest warrant for Bob.  The charging documents list Alice as Jane Doe.</p>
<p>At what point(s) in the proceedings can the public get access to the (possibly redacted) records for this case,
such as police report(s), witness statements, etc?</p>
<p>If Alice has given interviews to the press (as Jane Doe) about the case, would that change
the availability of the records?</p>

- Direct: none (query delta 0.0)
- Llama:  slot 10 delta -0.113 (query delta -0.1128)

Pair slot 8: incumbent law/ef2acbab_43073.txt (rel 0), challenger law/101678_72ec8248848560fa_279.txt (rel 0), true delta +0.000

INCUMBENT: For both private and Crown prosecutions, if process is issued, the documents of the pre- Enquête hearing become publicly accessible once the defendant has been arrested or the summons has been served, unless there are legislative restrictions to access (for example, the YCJA) or a court order restricts access.

CHALLENGER: **Daily News v. Bratton**, Supreme Court, Kings County, NYLJ, December 21, 1994 -- Newspaper sought investigative files relating to a rape. Columnist wrote several articles regarding the matter, and the victim, "Jane Doe", brought a libel action. Agency opposed disclosure, citing §50-b of the Civil Rights Law. Petitioner claimed that good cause for disclosure under §50-b was demonstrated due to its need for the records in defense civil litigation, while respondents claimed that records could be withheld currently under §87(2)(e). Court held that disclosure currently could prejudice the rights 

Pair slot 9: incumbent law/890af61b_43283.txt (rel 0), challenger law/101678_72ec8248848560fa_279.txt (rel 0), true delta +0.000

INCUMBENT: In general, statutory protections of a victim s right to privacy related to information contained in criminal justice records most often take the following forms: (1) prohibition against compelling testimony relating to personal information in open court; (2) exclusion or limited disclosure of victim identifying information in criminal justice records, including law enforcement reports, court materials, and prosecution documents; and (3) protection from release of addresses and phone numbers provided for notice purposes.

CHALLENGER: **Daily News v. Bratton**, Supreme Court, Kings County, NYLJ, December 21, 1994 -- Newspaper sought investigative files relating to a rape. Columnist wrote several articles regarding the matter, and the victim, "Jane Doe", brought a libel action. Agency opposed disclosure, citing §50-b of the Civil Rights Law. Petitioner claimed that good cause for disclosure under §50-b was demonstrated due to its need for the records in defense civil litigation, while respondents claimed that records could be withheld currently under §87(2)(e). Court held that disclosure currently could prejudice the rights 

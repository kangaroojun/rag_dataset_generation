from typing import List, Tuple

########################################
# Dataset Generation Utility Functions #
########################################

def format_evaluate_chunk_template(chunk: str) -> str:
    return f"""Given a chunk, complete the following task and return the result as a single integer of 0 or 1.
    Evaluate the supplied chunk and assign a numerical score of either 0 (Low) of 1 (High) for each of the 
    following criteria in your response:

    - **self_containment**: Evaluate whether the chunk provides enough information to be understood on its own, 
    without requiring additional external knowledge. If the chunk mentions an individual, or an organisation, the 
    identity of either entity must be clear in a self-contained chunk. A score of 1 indicates the chunk is complete 
    and self-contained, while a score of 0 reflects the need for further information.
    - **not_metadata**: Evaluate whether the chunk primarily consists of citations, links, dates, or other structural
    or metadata elements. A score of 0 indicates that the chunk contains primarily metadata, while a score of 1
    indicates that the chunk contains primarily content.

    **
    IMPORTANT: Please make sure to only return in dictionary format, with the 'self_containment' and 'not_metadata' keys. Be strict in your evaluation. It is better to have a lower score (0) than to be lenient in your evaluation.

    Example chunk: "Retrieved 2010-08-10. 35. T. Krovetz, W. Dai (2010). "How to get fast AES calls?" (https://groups.google.com/group/crypto pp-users/msg/a688203c2314ef08). Crypto++ user group. Retrieved 2010-08-11. 36. "Crypto++ 5.6.0 Pentium 4 Benchmarks" (http://www.cryptopp.com/benchmarks-p4.html). Crypto++ Website. 2009. Archived (https://web.archive.org/web/20100919121759/http://cryptop p.com/benchmarks-p4.html) from the original on 19 September 2010. Retrieved 2010-08-10."
    Example output: {{"self_containment": 1, "not_metadata": 0}}
    Reason: Large number of website urls, numbers and dates suggest that the chunk is primarily metadata.

    Example chunk: "5. Rivest, Ron L.; Shamir, Adi; Adleman, Len (September 20, 1983) [1977]. Cryptographic Communications System and Method. Cambridge MA. 4405829. 6. Brown, Bob (February 7, 2005). "Security's inseparable couple: Alice & Bob" (https://www.netw orkworld.com/article/2318241/lan-wan-security-s-inseparable-couple.html). NetworkWorld. 7. Rabin, Michael O. (1981). How to exchange secrets with oblivious transfer. Aiken Computation Lab, Harvard University. Technical Report TR-81. 8. Blum, Manuel (November 10, 1981). "Coin Flipping by Telephone a Protocol for Solving Impossible Problems" (https://doi.org/10.1145%2F1008908.1008911). ACM SIGACT News. 15 (1): 2327. doi:10.1145/1008908.1008911 (https://doi.org/10.1145%2F1008908.1008911). S2CID19928725 (https://api.semanticscholar.org/CorpusID:19928725). 9. Blum, Manuel (1983). "How to exchange (Secret) keys" (https://doi.org/10.1145%2F357360.35 7368). ACM Transactions on Computer Systems. 1 (2): 175193. doi:10.1145/357360.357368 ("
    Example output: {{"self_containment": 1, "not_metadata": 0}}
    Reason: Large number of website urls, numbers and dates suggest that the chunk is primarily metadata.

    Example chunk: "I loved music and thought I could be very good, but I knew I would never be John Coltrane or Stan Getz. I was interested in medicine and thought I could be a fine doctor, but I knew I would never be Michael DeBakey."
    Example output: {{"self_containment": 0, "not_metadata": 1}}
    Reason: Unclear who the individuals mentioned are, hence the chunk is not self-contained.

    Example chunk: "38. Attacks that show that the cipher does not perform as advertised (i.e., the level of difficulty involved in breaking it is lower than claimed), which are nevertheless of high enough complexity so that they are not practically achievable. 39. FIPS PUB 46-3 Data Encryption Standard (DES) (http://csrc.nist.gov/publications/fips/fips46-3/f ips46-3.pdf) (This is the third edition, 1999, but includes historical information in the preliminary section 12.) 40. NIST Special Publication 800-57 Recommendation for Key Management Part 1: General (Revised), March, 2007 (http://csrc.nist.gov/publications/nistpubs/800-57/sp800-57-Part1-revis ed2_Mar08-2007.pdf) Archived (https://web.archive.org/web/20140606050814/http://csrc.nist.g ov/publications/nistpubs/800-57/sp800-57-Part1-revised2_Mar08-2007.pdf) June 6, 2014, at the Wayback Machine. 41."
    Example output: {{"self_containment": 1, "not_metadata": 0}}
    Reason: Large number of website urls, numbers and dates suggest that the chunk is primarily metadata.

    Example chunk: "24. The Register, UK; Dan Goodin; 30 March 2008; Get your German Interior Minister's fingerprint, here. Compared to other solutions, "It's basically like leaving the password to your computer everywhere you go, without you being able to control it anymore", one of the hackers comments. (https://www.theregister.co.uk/2008/03/30/german_interior_minister_fingerprint_app ropriated) Archived (https://web.archive.org/web/20170810131615/https://www.theregister.co.u k/2008/03/30/german_interior_minister_fingerprint_appropriated) 10 August 2017 at the Wayback Machine 25. "Best Practices for Creating a Secure Guest Account" (https://technet.microsoft.com/en-us/libra ry/ff687018.aspx). 31 August 2016."
    Example output: {{"self_containment": 1, "not_metadata": 0}}
    Reason: Large number of website urls, numbers and dates suggest that the chunk is primarily metadata.

    Example chunk: "[309] The William J. Clinton Presidential Center and Park in Little Rock, Arkansas, was dedicated in 2004.[310] Clinton released a best-selling autobiography, My Life, in 2004.[311] In 2007, he released Giving: How Each of Us Can Change the World, which also became a New York Times Best Seller and garnered positive reviews.[312] In the aftermath of the 2004 Asian tsunami, U.N."
    Example output: {{"self_containment": 1, "not_metadata": 1}}
    Reason: The chunk is self-contained and does not contain metadata.

    Your output MUST only be a dictionary following the above format. Do not add any additional information to your response.
    **

    Chunk:
    {chunk}

    Output:
    """

def format_context_query_template(context: List[List[Tuple[str, str]]], chunks_per_context: int) -> str:
    context = "SEPARATOR \n".join([f"{chunk[1]}\n" for chunk in context])
    prompt = f"""You are a curious student who is great at asking inquisitive questions. Your task is to come up with
    a question based on the context information provided below. Note that the context provided
    is made up of {chunks_per_context} chunks (separated by the word "SEPARATOR"), and that the question generated 
    must consider all {chunks_per_context} chunks before generation. The questions should be self-contained and not require any external 
    knowledge to answer. Give only the questions, and no extra commentary, formatting, or chattiness.
    
    **
    IMPORTANT: The question generated should be unified and concise. It should not be multiple questions concatenated together.

    Example context:
    was a notable exception to the rest of Greece, ruled through the whole period by not one, but two hereditary monarchs. This was a form of diarchy. The Kings of Sparta belonged to the Agiads and the Eurypontids, descendants respectively of Eurysthenes and Procles. Both dynasties' founders were believed to be twin sons of Aristodemus, a Heraclid ruler. However, the powers of these kings were held in check by both a council of elders (the Gerousia) and magistrates specifically appointed to watch over the kings (the Ephors). Only free, land-owning, native-born men could be citizens entitled to the full protection of the law in a city-state. In most city-states, unlike the situation in Rome, social prominence did not allow special rights. Sometimes families controlled public religious functions, but this ordinarily did not give any extra power in the government. In Athens, the population was divided into four social classes based on wealth. People could change classes if they made more money. In Sparta, all male citizens were called homoioi, meaning "peers". However, Spartan kings, who served as the city- state's dual military and religious leaders, came from two families.[83] Slaves had no power or status. Slaves had the right to have a family and own property, subject to their master's goodwill and permission, but they had no political rights. By 600 BC, chattel slavery had spread in Greece. By the 5th century BC, slaves made up one-third of the total population in Social structure Slavery 09/10/2024, 17:42 Ancient Greece - Wikipedia https://en.wikipedia.org/wiki/Ancient_Greece 10/22
    SEPARATOR 
    adical solution to prevent the aristocracy regaining power. A citizens' assembly (the Ecclesia), for the discussion of city policy, had existed since the reforms of Draco in 621 BC; all citizens were permitted to attend after the reforms of Solon (early 6th century), but the poorest citizens could not address the assembly or run for office. With the establishment of the democracy, the assembly became the de jure mechanism of government; all citizens had equal privileges in the assembly. However, non-citizens, such as metics (foreigners living in Athens) or slaves, had no political rights at all. After the rise of democracy in Athens, other city-states founded democracies. However, many retained more traditional forms of government. As so often in other matters, Sparta was a notable exception to the rest of Greece, ruled through the whole period by not one, but two hereditary monarchs. This was a form of diarchy. The Kings of Sparta belonged to the Agiads and the Eurypontids, descendants respectively of Eurysthenes and Procles. Both dynasties' founders were believed to be twin sons of Aristodemus, a Heraclid ruler. However, the powers of these kings 
    SEPARATOR 
    Cleisthenes carried out further democratising reforms.[22] In Sparta, a political system with two kings, a council of elders, and five ephors developed over the course of the eighth and seventh century. According to Spartan tradition, this constitution was established by the legendary lawgiver Lycurgus.[23] Over the course of the first and second Messenian wars, Sparta subjugated the neighbouring region of Messenia, enserfing the population.[24] In the sixth century, Greek city-states began to develop formal relationships with one another, where previously individual rulers had relied on personal relationships with the elites of other cities.[25] Towards the end of the archaic period, Sparta began to build a series of alliances, the Peloponnesian League, with cities including Corinth, Elis, and Megara,[26] isolating Messenia and reinforcing Sparta's position against Argos, the other major power in the Peloponnese.[27] Other alliances in the sixth century included those between Elis and Heraea in the Peloponnese; and between the Greek colony Sybaris in southern Italy, its allies, and the Serdaioi.[28] In 499 BC, the Ionian city states under Persian rule rebelled against 

    Example positive query (Considers all chunks):
    "How did political structures, social hierarchies, and alliances shape the governance and interactions among Greek city-states, particularly focusing on Sparta and Athens?"

    Example negative query (Considers only last chunk, when should be considering all chunks):
    "What were the alliances that Sparta formed during the archaic period, and how did they help maintain its influence over Messenia?"

    Example context:
    Arabic. Abu Muhammad al-Hasan al-Hamdani had another view; he states that Arabs were called gharab ('westerners') by Mesopotamians because Bedouins originally resided to the west of Mesopotamia; the term was then corrupted into Arab. Yet another view is held by al-Masudi that the word Arab was initially applied to the Ishmaelites of the Arabah valley. In Biblical etymology, Arab (Hebrew: arvi) comes from the desert origin of the Bedouins it originally described (arava means 'wilderness'). The root -r-b has several additional meanings in Semitic languagesincluding 'west, sunset', 'desert', 'mingle', 'mixed', 'merchant' and 'raven'and are "comprehensible" with all of these having varying degrees of relevance to the emergence of the name. It is also possible that some forms were metathetical from -B-R, 'moving around' (Arabic: -B-R, 'traverse') and hence, it is alleged, 'nomadic'.[103] Arabic is a Semitic language that belongs to the Afroasiatic language family. The majority of scholars accept the "Arabian peninsula" has long been accepted as the original Urheimat (linguistic homeland) of the Semitic languages.[104][105][106][107] with some scholars investigating if its origins are in the Levant.[108] The ancient Semitic-speaking peoples lived in the ancient Near East, including the Levant, Mesopotamia, and the Arabian Peninsula from the 3rd millennium BCE to Origins 09/10/2024, 17:43 Arabs - Wikipedia https://en.wikipedia.org/wiki/Arabs 3/76
    SEPARATOR 
    an archaic form of Arabic in 328 CE using the Nabataean alphabet, which refers to Imru' al-Qays ibn 'Amr as 'King of all the Arabs'.[100][101] Herodotus refers to the Arabs in the Sinai, southern Palestine, and the frankincense region (Southern Arabia). Other Ancient-Greek historians like Agatharchides, Diodorus Siculus and Strabo mention Arabs living in Mesopotamia (along the Euphrates), in Egypt (the Sinai and the Red Sea), southern Jordan (the Nabataeans), the Syrian steppe and in eastern Arabia (the people of Gerrha). Inscriptions dating to the 6th century BCE in Yemen include the term 'Arab'.[102] The most popular Arab account holds that the word Arab came from an eponymous father named Ya'rub, who was supposedly the first to speak Arabic. Abu Muhammad al-Hasan al-Hamdani had another view; he states that Arabs were called gharab ('westerners') by Mesopotamians because Bedouins originally resided to the west of Mesopotamia; the term was then corrupted into Arab. Yet another view is held by al-Masudi that the word Arab was initially applied to the Ishmaelites of the Arabah valley. In Biblical etymology, Arab (Hebrew: arvi) comes from the 
    SEPARATOR 
    an, playing a vital role in trade between Mesopotamia, and the Mediterranean.[77] Other prominent tribes include Midian, Ad, and Thamud mentioned in the Bible and Quran. Later, in 900 BCE, the Qedarites enjoyed close relations with the nearby Canaanite and Aramaean states, and their territory extended from Lower Egypt to the Southern Levant.[78] From 1200 BCE to 110 BCE, powerful kingdoms emerged such as Saba, Lihyan, Minaean, Qataban, Hadhramaut, Awsan, and Homerite emerged in Arabia.[79] According to the Abrahamic tradition, Arabs are descendants of Abraham through his son Ishmael.[80] During classical antiquity, the Nabataeans established their kingdom with Petra as the capital in 300 BCE,[81] by 271 CE, the Palmyrene Empire with the capital Palmyra, led by Queen Zenobia, encompassed the Syria Palaestina, Arabia Petraea, and Egypt, as well as large parts of Anatolia.[82] The Arab Itureans inhabited Lebanon, Syria, and northern Palestine (Galilee) during the Hellenistic and Roman periods.[83] The Osroene and Hatran were Arab kingdoms in Upper Mesopotamia around 200 CE.[84] In 164 CE, the Sasanians recognized the Arabs as "Arbayistan", meaning "land of the 09/10/2024, 17:43 Arabs - Wikipedia https://en.wikipedia.org/wiki/Arabs 1/76

    Example positive query (Considers all chunks, concise and unified):
    "How did the origins, migrations, and cultural development of the Arabs shape their presence across ancient regions?"

    Example negative query (Considers only little bit of last chunk, when should be considering all chunks, and fails to utilise all provided contexts):
    "Where did the Nabataeans build their capital, and what was their role in ancient trade?"

    Example negative query (Obviously two separate questions that are just concatenated together):
    "What are the origins and historical migrations of the Arabs, and how did their linguistic, cultural, and political influence evolve in ancient times?"
    **

    Context:
    {context}
    """
    return prompt

def format_chunk_query_template(chunk: str) -> str:
    return f"""You are a curious student who is great at asking inquisitive questions. Your task is to come up with
    a question based on the context information provided below. The question should be self-contained and not require any
    external knowledge to answer, and should not require referring to the context to know the topic of the question. There
    should be no mention of "as per the context provided" or any similar phrases in the output. Give only the question, 
    and no extra commentary, formatting, or chattiness.

    **

    Example context:
    "C++ supports function, class, alias, and variable templates. Templates may be parameterized by types, compile-time constants, and other templates. Templates are implemented by instantiation at compile-time."

    Example positive query (Can be answered by context):
    "How do templates contribute to programming in C++?"

    Example negative query (Do not know the topic of the question without looking at context):
    "What templates are there?"

    Example context:
    "In the case of a journal reference, VVVV is the volume number, M indicates the section of the journal where the reference was published (e.g., L for a letters section), PPPP gives the starting page number, and A is the first letter of the last name of the first author. Periods (.)"
    
    Example positive query (Can be answered by context):
    "For journal references, what do the different components like volume, section, page number, and author initials represent?"

    Example negative query (Do not know the topic of the question without looking at context, uses term "as per the context provided"):
    "As per the context provided, how does one reference author initials?"

    Example context:
    "The invasion is known to have displaced population to the later Attic-Ionic regions, who regarded themselves as descendants of the population displaced by or contending with the Dorians. The Greeks of this period believed there were three major divisions of all Greek people Dorians, Aeolians, and Ionians (including Athenians), each with their own defining and distinctive dialects."

    Example positive query (Can be answered by context):
    "Did the Dorians, Aeolians, and Ionians speak different dialects?"

    Example negative query (References context within question through phrase "during the period discussed", which is not allowed as reader does not have access to context):
    "What were the three major divisions of Greek people during the period discussed?"
    **

    Chunk:
    {chunk}
    """

def format_answer_query_template(query: str, chunks: List[str]) -> str:
    chunk_strings = "\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""You will be given a query and chunks that contain information relevant to the query. Your task is to:
    1. **Answer the query** based on the information provided in the chunks.

    The answer should touch on all the key points mentioned in the chunks. All chunks should be relevant to the answer.

    **

    Example Input:
    Query: "How did the concept of authentication evolve to include elements of knowledge, ownership, and validation across various fields and technologies?"
    Chunk 1: "or verify a person's identity before being granted access, approving a transaction request, signing a document or other work product, granting authority to others, and establishing a chain of authority. Security research has determined that for a positive authentication, elements from at least two, and preferably all three, factors should be verified.[6] The three factors (classes) and some of the elements of each factor are: 1. Knowledge: Something the user knows (e.g., a password, partial password, passphrase, personal identification number (PIN), challengeresponse (the user must answer a question or pattern), security question)."
    Chunk 2: "However, text, audio, and video can be copied into new media, possibly leaving only the informational content itself to use in authentication. Various systems have been invented to allow authors to provide a means for readers to reliably authenticate that a given message originated from or was relayed by them. These involve authentication factors like: A difficult-to-reproduce physical artifact, such as a seal, signature, watermark, special stationery, or fingerprint. A shared secret, such as a passphrase, in the content of the message."
    Chunk 3: "2. Ownership: Something the user has (e.g., wrist band, ID card, security token, implanted device, cell phone with a built-in hardware token, software token, or cell phone holding a software token). 3."
    Chunk 4: "[1] It might involve validating personal identity documents, verifying the authenticity of a website with a digital certificate,[2] determining the age of an artifact by carbon dating, or ensuring that a product or document is not counterfeit. Authentication is relevant to multiple fields. In art, antiques, and anthropology, a common problem is verifying that a given artifact was produced by a certain person or in a certain place or period of history. In computer science, verifying a user's identity is often required to allow access to confidential data or systems.[3] Authentication can be considered to be of three types: The first type of authentication is accepting proof of identity given by a credible person who has first-hand evidence that the identity is genuine. When authentication is required of art or physical objects, this proof could be a friend, family member, or colleague attesting to the item's provenance, perhaps by having witnessed the item in its creator's possession. With autographed sports memorabilia, this could involve someone attesting that they witnessed the object being signed."
    Chunk 5: "The European Central Bank (ECB) has defined strong authentication as "a procedure based on two or more of the three authentication factors". The factors that are used must be mutually independent and at least one factor must be "non-reusable and non-replicable", except in the case of an inherence factor and must also be incapable of being stolen off the Internet. In the European, as well as in the US-American understanding, strong authentication is very similar to multi-factor authentication or 2FA, but exceeding those with more rigorous requirements.[2][8] The FIDO Alliance has been striving to establish technical specifications for strong authentication.[9] Conventional computer systems authenticate users only at the initial log-in session, which can be the cause of a critical security flaw."

    Output: "The concept of authentication has evolved significantly over time to include three key elements: knowledge, ownership, and validation, which have been applied across various fields and technologies. Initially, authentication systems relied heavily on knowledge-based factors, such as passwords, PINs, and security questions. Over time, ownership-based factors were introduced, including physical items like ID cards, security tokens, or devices. In different fields, authentication also involves validating objects or identities. For instance, in computer science, verifying user identity is crucial for accessing systems, while in fields like art and anthropology, authentication involves validating the provenance of artifacts (Chunk 4). Additionally, physical artifacts such as seals or signatures and shared secrets like passphrases have been historically used for authentication. Modern authentication systems, such as multi-factor authentication (MFA), combine these elements to strengthen security by requiring two or more independent factors. Strong authentication procedures require that at least one factor be non-reusable and non-replicable, further increasing the reliability of the process."

    Example Input:
    Query: "What is a block cipher and how does it enhance security in cryptography?"
    Chunk 1: "Block cipher (Redirected from Block ciphers) In cryptography, a block cipher is a deterministic algorithm that operates on fixed-length groups of bits, called blocks. Block ciphers are the elementary building blocks of many cryptographic protocols. They are ubiquitous in the storage and exchange of data, where such data is secured and authenticated via encryption. A block cipher uses blocks as an unvarying transformation. Even a secure block cipher is suitable for the encryption of only a single block of data at a time, using a fixed key. A multitude of modes of operation have been designed to allow their repeated use in a secure way to achieve the security goals of confidentiality and authenticity. However, block ciphers may also feature as building blocks in other cryptographic protocols, such as universal hash functions and pseudorandom number generators. A block cipher consists of two paired algorithms, one for encryption, E, and the other for decryption, D."

    Output: "A block cipher is a deterministic algorithm in cryptography that processes data in fixed-length groups of bits, known as blocks. It plays a crucial role in securing and authenticating data during storage and transmission. Block ciphers function by transforming blocks of data using a fixed key, with separate algorithms for encryption (E) and decryption (D). Block ciphers enhance security in cryptography by allowing data to be encrypted in ways that ensure both confidentiality and authenticity. While a block cipher can only encrypt one block of data at a time with a single key, modes of operation are employed to extend their usage securely across multiple blocks. These modes ensure that the encryption remains secure and resistant to attacks, even when applied to large datasets. Additionally, block ciphers are used as components in other cryptographic protocols, including universal hash functions and pseudorandom number generators, further contributing to the robustness of cryptographic systems."

    Example Input:
    Query: "Why did Bill Clinton's signing of the Defense of Marriage Act (DOMA) in 1996 draw criticism, and how did his stance on gay rights evolve during his presidency?"
    Chunk 1: "Supreme Court struck down DOMA in June 2013.[142] Despite DOMA, Clinton was the first president to select openly gay persons for administrative positions,[143] and he is generally credited as being the first president to publicly champion gay rights.[144] During his presidency, Clinton issued two substantially controversial executive orders on behalf of gay rights, the first lifting the ban on security clearances for LGBT federal employees[145] and the second outlawing discrimination based on sexual orientation in the federal civilian workforce.[146] Under Clinton's leadership, federal funding for HIV/AIDS research, 09/10/2024, 17:56 Bill Clinton - Wikipedia https://en.wikipedia.org/wiki/Bill_Clinton 14/61"
    Chunk 2: "[133][134][135] Credit for this decline would be given to the growing effectiveness of new drug therapy which was promoted by the Clinton Administration's Department of Health and Human Services, such as highly active antiretroviral therapy (HAART).[134][135] On September 21, 1996, Clinton signed into law the Defense of Marriage Act (DOMA), which defined marriage for federal purposes as the legal union of one man and one woman; the legislation allowed individual states to refuse to recognize gay marriages that were performed in other states.[136] Paul Yandura, speaking for the White House gay and lesbian liaison office, said Clinton's signing DOMA "was a political decision that they made at the time of a re-election"."
    Chunk 3: "He also condemned homophobia and discrimination against people with HIV. Clinton announced three new initiatives: creating a special working group to coordinate AIDS research throughout the federal government; convening public health experts to develop an action plan that integrates HIV prevention with substance abuse prevention; and launching a new effort by the Department of Justice to ensure that health care facilities provide equal access to people with HIV and AIDS.[132] 1996 would mark the first year since the beginning of the HIV/AIDS epidemic that the number of new HIV/AIDS diagnoses would decline, with the U.S. Centers for Disease Control and Prevention (CDC) even later reporting a significant 47% decline in the number of AIDS-related deaths in 1997 compared to the previous year."
    Chunk 4: "In defense of his actions, Clinton has said that DOMA was intended to "head off an attempt to send a constitutional amendment banning gay marriage to the states", a possibility he described as highly likely in the context of a "very reactionary Congress".[137] Administration spokesman Richard Socarides said, "the alternatives we knew were going to be far worse, and it was time to move on and get the president re-elected."[138] Clinton himself said DOMA was something "which the Republicans put on the ballot to try to get the base vote for Bush up, I think it's obvious that something had to be done to try to keep the Republican Congress from presenting that";[139] others were more critical. The veteran gay rights and gay marriage activist Evan Wolfson has called these claims "historic revisionism".[138] Despite this, it has been noted that other than a brief written response to a Reader's Digest that questioned whether he agreed with it, Clinton had made no documented reference to the issue of gay marriage until May 1996.[140] In a July 2, 2011, editorial The New York Times opined, "The Defense of Marriage Act was enacted in 1996 as an election-year wedge issue, signed by President Bill Clinton in one of his worst policy moments."[141] Ultimately, in United States v. Windsor, the U.S."
    
    Output: "During his presidency, Bill Clinton's signing of the Defense of Marriage Act (DOMA) in 1996 drew significant criticism, despite his otherwise progressive actions on gay rights. DOMA defined marriage at the federal level as a union between one man and one woman and allowed states to refuse recognition of same-sex marriages performed elsewhere. Clinton later defended the signing of DOMA as a political move to prevent a potential constitutional amendment banning gay marriage, which he feared the Republican-led Congress would push forward. Clinton’s spokesperson Richard Socarides emphasized that rejecting DOMA could have led to worse outcomes for the LGBTQ+ community, describing the signing as a strategic decision during a tough re-election campaign. Despite the controversy surrounding DOMA, Clinton took several steps to advance gay rights during his time in office. He was the first president to appoint openly gay individuals to federal administrative positions and issued executive orders prohibiting discrimination against LGBT federal employees. Clinton also lifted the ban on security clearances for LGBT employees, and his administration increased federal funding for HIV/AIDS research. Critics viewed Clinton’s signing of DOMA as a political compromise. Some, such as gay rights activist Evan Wolfson, dismissed Clinton’s justification as "historic revisionism." Publications like The New York Times later described it as one of Clinton’s worst policy moments, asserting that it was used as an election-year wedge issue. Ultimately, Clinton’s actions reflect a mixed legacy on gay rights: while he made historic advances for LGBTQ+ representation and protections, his signing of DOMA remains a controversial and heavily criticized part of his presidency."
    
    IMPORTANT: You MUST return the output only in string format. Do not add any additional information to your response. Do not include any additional formatting or commentary.

    Query: {query}
    {chunk_strings}
    """

def format_separating_multi_query_template(query: str, chunks: List[str]) -> str:
    chunk_strings = "\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks)])
    return f"""You will be given a query that contains two questions. Your task is to:
    1. **Split the query** into the two individual questions that are self-contained (no pronouns to be included in the questions).
    2. **Identify the chunks** required to answer each question based on the provided list of chunks.
    3. **Return a JSON object** where each question is a key, and the value is a list of the chunks (referred to by their index) required to answer that question.

    If a chunk is not necessary to answer a particular question, it should not be included in the list for that question.
    All chunks NEED to be mapped.

    **

    IMPORTANT: Please make sure to only return in dictionary format.

    Example Input:
    Query: "How does binary counting work, and how does it differ from the decimal counting system?"
    Chunk 1: "Beginning with a single digit, counting proceeds through each symbol, in increasing order. Before examining binary counting, it is useful to briefly discuss the more familiar decimal counting system as a frame of reference. Decimal counting uses the ten symbols 0 through 9."
    Chunk 2: "Decimal number Binary number 0 0 1 1 2 10 3 11 4 100 5 101 6 110 7 111 8 1000 9 1001 10 1010 11 1011 12 1100 13 1101 14 1110 15 1111 This counter shows how to count in binary from numbers zero through thirty-one. 0b100101 (a prefix indicating binary format, common in programming languages) 6b100101 (a prefix indicating number of bits in binary format, common in programming languages) #b100101 (a prefix indicating binary format, common in Lisp programming languages) When spoken, binary numerals are usually read digit-by-digit, to distinguish them from decimal numerals. For example, the binary numeral 100 is pronounced one zero zero, rather than one hundred, to make its binary nature explicit and for purposes of correctness."
    Chunk 3: "Since the binary numeral 100 represents the value four, it would be confusing to refer to the numeral as one hundred (a word that represents a completely different value, or amount). Alternatively, the binary numeral 100 can be read out as "four" (the correct value), but this does not make its binary nature explicit. Counting in binary is similar to counting in any other number system."
    Chunk 4: "For example, the binary number 100101 is converted to decimal form as follows: 1001012 = [ ( 1 ) 25 ] + [ ( 0 ) 24 ] + [ ( 0 ) 23 ] + [ ( 1 ) 22 ] + [ ( 0 ) 21 ] + [ ( 1 ) 20 ] 1001012 = [ 1 32 ] + [ 0 16 ] + [ 0 8 ] + [ 1 4 ] + [ 0 2 ] + [ 1 1 ] 1001012 = 3710 Fractions in binary arithmetic terminate only if the denominator is a power of 2. As a result, 1/10 does not have a finite binary representation (10 has prime factors 2 and 5)."
    Chunk 5: "Arithmetic values thought to have been represented by parts of the Eye of Horus Binary number (Redirected from Binary numeral system) A binary number is a number expressed in the base-2 numeral system or binary numeral system, a method for representing numbers that uses only two symbols for the natural numbers: typically "0" (zero) and "1" (one). A binary number may also refer to a rational number that has a finite representation in the binary numeral system, that is, the quotient of an integer by a power of two. The base-2 numeral system is a positional notation with a radix of 2."

    Output: {{"How does binary counting work?": [2, 3, 4, 5], "How does binary counting differ from the decimal counting system?": [1]}}
    Reason: The original query contains two distinct questions: 1) "How does binary counting work?" and 2) "How does it differ from the decimal counting system?" Query 1 evidently requires Chunks 2-5 to be answered, while query 2 requires only chunk 1 to be answered.

    Example Input:
    Query: "What key features prompt cryptographic primitive evaluation, and how does the RC5 block cipher algorithm utilize data-dependent rotations, modular additions, and XORs in its design?"
    Chunk 1: "The encryption and decryption routines can be specified in a few lines of code. The key schedule, however, is more complex, expanding the key using an essentially one-way function with the binary expansions of both e and the golden ratio as sources of "nothing up my sleeve numbers". The tantalizing simplicity of the algorithm together with the novelty of the data-dependent rotations has made RC5 an attractive object of study for cryptanalysts. 12-round RC5 (with 64-bit blocks) is susceptible to a differential attack using 244 chosen plaintexts.[41] 1820 rounds are suggested as sufficient protection. IDEA RC5 Rijndael / AES 09/10/2024, 17:47 Block cipher - Wikipedia https://en.wikipedia.org/wiki/Block_cipher 10/15"
    Chunk 2: "As of 2012, the best attack which applies to all keys can break a full 8.5-round IDEA using a narrow-bicliques attack about four times faster than brute force. RC5 is a block cipher designed by Ronald Rivest in 1994 which, unlike many other ciphers, has a variable block size (32, 64, or 128 bits), key size (0 to 2040 bits), and a number of rounds (0 to 255). The original suggested choice of parameters was a block size of 64 bits, a 128-bit key, and 12 rounds. A key feature of RC5 is the use of data-dependent rotations; one of the goals of RC5 was to prompt the study and evaluation of such operations as a cryptographic primitive. RC5 also consists of a number of modular additions and XORs."
    Chunk 3: "One round (two half- rounds) of the RC5 block cipher machine designed to break DES was demonstrated in 1998 by the Electronic Frontier Foundation. An extension to DES, Triple DES, triple-encrypts each block with either two independent keys (112-bit key and 80-bit security) or three independent keys (168-bit key and 112-bit security). It was widely adopted as a replacement."
    Chunk 4: "A round can then be performed with 16 table lookup operations and 12 32-bit exclusive-or operations, followed by four 32-bit exclusive-or operations in the AddRoundKey step.[12] Alternatively, the table lookup operation can be performed with a single 256-entry 32-bit table (occupying 1024 bytes) followed by circular rotation operations. Using a byte-oriented approach, it is possible to combine the SubBytes, ShiftRows, and MixColumns steps into a single round operation.[13] The National Security Agency (NSA) reviewed all the AES finalists, including Rijndael, and stated that all of them were secure enough for U.S."
    Chunk 5: "Most block cipher algorithms are classified as iterated block ciphers which means that they transform fixed-size blocks of plaintext into identically sized blocks of ciphertext, via the repeated application of an invertible transformation known as the round function, with each iteration referred to as a round.[12] Usually, the round function R takes different round keys Ki as a second input, which is derived from the original key: where is the plaintext and the ciphertext, with r being the number of rounds. History Design Iterated block ciphers 09/10/2024, 17:47 Block cipher - Wikipedia https://en.wikipedia.org/wiki/Block_cipher 2/15"

    Output: {{"What key features prompt cryptographic primitive evaluation?": [1, 2, 3, 5], "How does the RC5 block cipher algorithm utilize data-dependent rotations, modular additions, and XORs in its design?": [2, 3, 4]}}
    Reason: The original query contains two distinct questions: 1) "What key features prompt cryptographic primitive evaluation?" and 2) "How does the RC5 block cipher algorithm utilize data-dependent rotations, modular additions, and XORs in its design?" Chunks 1, 2, 3, and 5 are essential for understanding the key features prompting cryptographic primitive evaluation. They cover the novelty of RC5, cryptanalysis efforts, and fundamental concepts related to block ciphers. Chunks 2, 3, and 4 provide insight into the design of RC5 and how it employs data-dependent rotations, modular additions, and XORs.

    Do not output the reason. Only return the dictionary with the 'query' and 'chunks_to_remove' keys. The query selected 
    should result in more chunks retained than removed. 
    
    IMPORTANT: All chunks MUST be mapped to a question.

    Query: {query}
    {chunk_strings}
    """

#########################################
# Helper Functions for Evolving Queries #
#########################################

base_instruction = """I want you to act as an input rewriter.
    Your object is the rewrite a given `input` and must be factually correct according to the supporting information in `Context`.
    You MUST complicate the given `Input` using the following method:"""

def multi_context_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. `Input` should be rewritten to require readers to use information from all elements of `Context`. 
        2. `Rewritten Input` must be fully answerable from information in `Context`. 
        3. `Rewritten Input` should be concise and understandable by humans.
        4. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
        5. `Rewritten Input` should not contain more than 20 words. Use abbreviation wherever possible.
        
        **
        EXAMPLES

        Example context:
        ["Vaccines introduce a weakened or dead form of the pathogen to the human body.", "This exposure helps the immune system learn to recognize and fight the pathogen in the future."]
        Example input:
        How do vaccines work?
        Example rewritten input:
        How does the introduction of a modified pathogen prepare the immune system for future encounters?

        --------------------------
        
        Example context:
        ["Plants perform photosynthesis, using sunlight to convert carbon dioxide and water into glucose and oxygen.", "Chlorophyll in plant leaves absorbs sunlight, initiating the photosynthesis process.", "Oxygen is a by-product of the photosynthesis process and is released into the atmosphere."]
        Example input:
        Explain how plants produce oxygen.
        Example rewritten input: 
        Considering chlorophyll's role in sunlight absorption and photosynthesis, how is oxygen produced and released by plants?

        --------------------------

        Example context:
        ["The gravitational pull of the moon on the Earth influences the tides.", "The position of the sun relative to the Earth and the moon also affects tidal patterns."]
        Example input:
        Tell me about high tides.
        Example rewritten input:
        Explain how the combined gravitational effects of the moon and the sun's relative positioning influence Earth's tidal phenomena.
        

        --------------------------

        Example single long context:
        “A machine learning model is trained on a dataset that represents real-world examples. This training process helps the model learn 
        patterns and make predictions when encountering new, unseen data in the future.”
        Example input:
        What does a machine learning model learn from a dataset?
        Example rewritten input:
        How does training a machine learning model on a dataset help the model make accurate predictions on unseen data?
        **

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:            
        """
    )

def reasoning_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. If `Input` can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
        2. `Rewritten Input` should require readers to make multiple logical connections or inferences.
        3. `Rewritten Input` should be concise and understandable by humans.
        4. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
        5. `Rewritten Input` must be fully answerable from information in `Context`. 
        6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES

        Example context:
        Chlorophyll allows plants to absorb energy from light, and this energy is used to convert carbon dioxide and water into glucose and oxygen, a process known as photosynthesis.
        Example input:
        Why are plants green?
        Example rewritten input:
        How does chlorophyll's role in absorbing light relate to plants' green color and their ability to produce glucose?
    
        --------------------------
        
        Example context:
        The greenhouse effect occurs when the Earth's atmosphere traps solar radiation, caused by gases such as carbon dioxide, methane, and water vapor. This process maintains the planet's temperature but can lead to increased global temperatures when exacerbated by human activities.
        Example input:
        What causes seasons to change?
        Example rewritten input: 
        Given the trapping of solar radiation by atmospheric gases, explain how the enhanced activity impact Earth's climate.

        --------------------------

        Example context:
        Economic theories suggest that market demand and supply determine prices, but government policies can also influence market dynamics through regulations, taxes, and subsidies.
        Example input:
        Identify the primary factors that determine the price of goods in a market.
        Example rewritten input:
        Examine how the interplay of market demand, supply dynamics, and government policy interventions collectively shape the pricing mechanism of goods within a market ecosystem.
        **

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:            
        """
    )

def concretizing_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` by replacing general concepts/inquiries with more specific ones.
        2. `Rewritten Input` should be concise and understandable by humans.
        3. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
        4. `Rewritten Input` must be fully answerable from information in `Context`.  
        5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES
        Example context:
        Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
        The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
        Example input: 
        Why is the biodiversity of rainforests important?
        Example rewritten input:
        How does the extensive biodiversity found in rainforests, encompassing over half of the world's plant and animal species, contribute to global biodiversity maintenance, and what role does this diversity play in enhancing ecosystem resilience, human health through disease control, crop pollination, and the development of medicines derived from rainforest plants?

        --------------------------

        Example context:
        Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
        Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
        Example input: 
        What is the role of bees in ecosystems?
        Example rewritten input:
        How do bees, through their pollination of flowering plants, including a multitude of fruits and vegetables, significantly influence the diversity of plant life and agricultural productivity, and in what ways do their activities extend beyond agricultural settings to support the growth of trees, flowers, and other plants, thereby providing essential resources for various animal species and contributing to the overall balance and sustainability of ecosystems?

        --------------------------

        Example context:
        Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
        Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
        Example input: 
        What are the principles behind solar power generation?
        Example rewritten input:
        How do photovoltaic cells work to convert sunlight into electrical power, and what role do solar panels play in this process, including energy storage for sustainable use?
        **

        Input:
        {input}
        Context:
        {context}
        Rewritten Input:
        """
    )
    
def generalizing_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` by removing specific details and replacing them with general concepts.
        2. Ensure `Rewritten Input` is broad, simple, and invites a wide range of possible answers.
        3. `Rewritten Input` should not contain any overly specific terms or technical jargon.
        4. Keep `Rewritten Input` concise, aiming for open-ended questions that do not exceed 10 words.
        
        **
        EXAMPLES
        Example context:
        Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances. The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
        Example input: 
        How does the extensive biodiversity found in rainforests, encompassing over half of the world's plant and animal species, contribute to global biodiversity maintenance, and what role does this diversity play in enhancing ecosystem resilience, human health through disease control, crop pollination, and the development of medicines derived from rainforest plants?
        Example rewritten input:
        Why are rainforests important?

        --------------------------

        Example context:
        Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance. Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
        Example input: 
        How do bees, through their pollination of flowering plants, including a multitude of fruits and vegetables, significantly influence the diversity of plant life and agricultural productivity, and in what ways do their activities extend beyond agricultural settings to support the growth of trees, flowers, and other plants, thereby providing essential resources for various animal species and contributing to the overall balance and sustainability of ecosystems?
        Example rewritten input:
        Why are bees important in ecosystems?

        --------------------------

        Example context:
        Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current. Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
        Example input: 
        How do photovoltaic cells work to convert sunlight into electrical power, and what role do solar panels play in this process, including energy storage for sustainable use?
        Example rewritten input:
        How does solar power work?

        **

        Input:
        {input}
        Context:
        {context}
        Rewritten Input:
        """
    )

def constrained_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` by adding at least one more constraints/requirements.
        2. `Rewritten Input` must be fully answerable from information in `Context`. 
        5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES
        Example context:
        Rainforests are home to over half of the world's plant and animal species, making them key to maintaining global biodiversity. The variety of life found in these ecosystems contributes to genetic diversity, which is crucial for adaptation and survival amid changing environmental conditions. This biodiversity also supports ecosystem resilience, enabling forests to recover from disturbances.
        The biodiversity in rainforests plays a significant role in human well-being, providing essential services such as air and water purification, disease control, and pollination of crops. Additionally, many medicines are derived from rainforest plants, highlighting the importance of these ecosystems for medical research and healthcare.
        Example input: 
        Why is the biodiversity of rainforests important?
        Example rewritten input:
        How does the biodiversity of rainforests contribute to ecosystem resilience and recovery from disturbances, and in what ways does it impact human well-being through services such as air and water purification, disease control, and crop pollination?

        --------------------------

        Example context:
        Bees play a critical role in pollinating flowering plants, including many fruits and vegetables, contributing to the diversity of plant life and the production of crops. Their activity supports the growth of trees, flowers, and other plants, which serve as food and shelter for numerous animals, thus maintaining ecosystem balance.
        Beyond their impact on food crops, bees contribute to wild plant growth by pollinating a wide range of plants outside of agricultural settings. This pollination is vital for the reproduction of many plants, affecting entire ecosystems' health and sustainability.
        Example input: 
        What is the role of bees in ecosystems?
        Example rewritten input:
        Considering the pivotal role bees play in pollinating both agricultural crops and wild plants, thereby contributing to the diversity of plant life and supporting the foundation of food chains, analyze how bees influence the growth and sustainability of various ecosystems.

        --------------------------

        Example context:
        Solar power generation relies on photovoltaic cells to convert sunlight into electricity. These cells are made of materials that exhibit the photovoltaic effect, which occurs when light photons are absorbed by the material, causing the generation of electrical current.
        Solar panels, composed of many photovoltaic cells, collect sunlight and convert it into electrical power. This energy can then be used directly or stored in batteries for later use, providing a renewable and sustainable source of power with minimal environmental impact.
        Example input: 
        What are the principles behind solar power generation?
        Example rewritten input:
        Examine the significance of rainforest biodiversity in sustaining ecosystem resilience and providing essential services such as disease control and crop pollination, alongside its critical role in medical research and the development of new medicines. Consider the broader implications of biodiversity loss on global ecological balance and human health.
        **

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:
        """
    )

def comparative_question_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` to focus on comparing two or more entities, concepts, or processes.
        2. `Rewritten Input` should encourage a detailed comparison that highlights similarities and differences.
        3. `Rewritten Input` must be fully answerable from information in `Context`. 
        4. `Rewritten Input` should be concise and understandable by humans.
        5. `Rewritten Input` should not contain phrases like  'based on the provided context' or 'according to the context'.
        6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES
        Example context:
        "Water boils at 100°C (212°F) at sea level, but boiling point decreases with altitude due to lower atmospheric pressure. In contrast, alcohol boils at about 78°C (172°F)."
        Example input: 
        What happens to water as it boils?
        Example rewritten input:
        How does the boiling point of water at sea level compare to that of alcohol, and how does altitude affect water's boiling point?

        --------------------------

        Example context:
        "Photosynthesis in plants involves converting carbon dioxide and water into glucose and oxygen, using sunlight. Cellular respiration in animals converts glucose and oxygen back into carbon dioxide and water, releasing energy."
        Example input: 
        How do plants and animals process energy?
        Example rewritten input:
        Compare the processes of photosynthesis in plants and cellular respiration in animals, focusing on inputs and outputs of each process.

        --------------------------

        Example context:
        "The Renaissance was a period of significant cultural, artistic, and scientific rebirth that began in the 14th century, primarily in Italy. The Enlightenment, occurring mainly in the 18th century, centered around reason, science, and individualism, significantly influencing European thought."
        Example input: 
        What was the Renaissance?
        Example rewritten input:
        Contrast the main focuses and impacts of the Renaissance and the Enlightenment on European thought and culture.

        --------------------------

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:
        """
    )

def hypothetical_scenario_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` to include a hypothetical or speculative scenario that is relevant to the `Context`.
        2. `Rewritten Input` should encourage the reader to apply knowledge from the `Context` to imagine or deduce outcomes.
        3. `Rewritten Input` should be concise, clear, and understandable by humans.
        4. `Rewritten Input` should not contain phrases like 'based on the provided context' or 'according to the context'.
        5. `Rewritten Input` must be fully answerable from information in `Context`.
        6. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES

        Example context:
        The greenhouse effect is a natural process where the Earth's atmosphere traps some of the Sun's energy, warming the planet to a temperature that supports life. Human activities, particularly the emission of greenhouse gases like carbon dioxide and methane, have intensified this effect, leading to global warming and climate change.
        Example input:
        What are the consequences of the greenhouse effect?
        Example rewritten input:
        Imagine a world where greenhouse gas emissions were doubled overnight. How might this intensified greenhouse effect impact global climate patterns and ecosystems?

        --------------------------

        Example context:
        Antibiotics are drugs used to treat bacterial infections. They work by killing bacteria or preventing their growth. However, overuse and misuse of antibiotics have led to the development of antibiotic-resistant bacteria, which are harder to treat because they can withstand the drugs designed to kill them.
        Example input:
        How do antibiotics work?
        Example rewritten input:
        In a scenario where a new antibiotic-resistant superbug emerges, how would the principles of antibiotic action and resistance influence our approach to treatment?

        --------------------------

        Example context:
        Quantum computing relies on the principles of quantum mechanics to process information, utilizing quantum bits or qubits. These qubits can exist in multiple states simultaneously, allowing quantum computers to perform complex calculations much faster than traditional computers.
        Example input:
        What is quantum computing?
        Example rewritten input:
        Suppose a quantum computer was tasked with solving a problem that currently takes traditional computers centuries to solve. How might the unique capabilities of quantum computing change the outcome?
        **

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:
        """
    )

def in_breadth_evolution(input, context):
    return (
        base_instruction
        + f"""
        1. Rewrite `Input` to create a create a brand new prompt.
        2. `Rewritten Input` should belong to the same domain as the `input` but be even more rare.
        3. `Rewritten Input` should be concise, clear, and understandable by humans.
        4. `Rewritten Input` should not contain phrases like 'based on the provided context' or 'according to the context'.
        5. `Rewritten Input` should not contain more than 15 words. Use abbreviation wherever possible.

        **
        EXAMPLES

        Example context:
        Wearable technology has revolutionized personal health monitoring, allowing individuals to track vital signs and activity levels in real time.
        Example input:
        Explore the impact of wearable technology on personal health management.
        Example rewritten input:
        Delve into the development of implantable health devices and their potential to transform chronic disease management.

        --------------------------

        Example context:
        Quantum computing leverages the principles of quantum mechanics to process information, offering significant advancements over traditional computing methods.
        Example input:
        How is quantum computing different from traditional computing?
        Example rewritten input:
        Explore the potential of quantum cryptography in enhancing cybersecurity measures beyond current encryption standards

        --------------------------

        Example context:
        Virtual reality (VR) offers immersive learning experiences, transforming educational methodologies by providing interactive and engaging ways to acquire knowledge, especially in fields requiring practical skills.
        Example input:
        What impact does virtual reality (VR) have on education?
        Example rewritten input:
        Investigate the use of VR simulations in medical training to enhance practical skills and decision-making under pressure.
        **

        Context:
        {context}
        Input:
        {input}
        Rewritten Input:
        """
    )
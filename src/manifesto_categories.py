"""
This module contains the category definitions for the Manifesto Project coding scheme.
Each category has a unique ID, title, description, and placeholders for result and explanation.
Categories are grouped by series (100s for external relations, 200s for freedom and democracy, etc.).
"""

ALL_CATEGORIES = {
    "101": {
        "title": "Foreign Special Relationships: Positive",
        "description": "Favourable mentions of particular countries with which their country has a special relationship; the need for co-operation with and/or aid to such countries.",
        "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "102": {
    "title": "Foreign Special Relationships: Negative",
    "description": "Negative mentions of particular countries with which their country has a special relationship.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "103": {
    "title": "Anti-Imperialism",
    "description": "Negative references to imperial behaviour and/or negative references to one state exerting strong influence (political, military or commercial) over other states. May also include: Negative references to controlling other countries as if they were part of an empire; Favourable references to greater self-government and independence for colonies; Favourable mentions of de-colonisation.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "104": {
    "title": "Military: Positive",
    "description": "The importance of external security and defence. May include statements concerning: The need to maintain or increase military expenditure; The need to secure adequate manpower in the military; The need to modernise armed forces and improve military strength; The need for rearmament and self-defence; The need to keep military treaty obligations.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "105": {
    "title": "Military: Negative",
    "description": "Negative references to the military or use of military power to solve conflicts. References to the ‘evils of war’. May include references to: Decreasing military expenditures; Disarmament; Reduced or abolished conscription.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "106": {
    "title": "Peace",
    "description": "Any declaration of belief in peace and peaceful means of solving crises -- absent reference to the military. May include: Peace as a general goal; Desirability of countries joining in negotiations with hostile countries; Ending wars in order to establish peace.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "107": {
    "title": "Internationalism: Positive",
    "description": "Need for international co-operation, including co-operation with specific countries other than those coded in 101. May also include references to the: Need for aid to developing countries; Need for world planning of resources; Support for global governance; Need for international courts; Support for UN or other international organisations.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "108": {
    "title": "European Community/Union: Positive",
    "description": "Favourable mentions of European Community/Union in general. May include the: Desirability of their country joining (or remaining a member); Desirability of expanding the European Community/Union; Desirability of increasing the ECs/EUs competences; Desirability of expanding the competences of the European Parliament.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },    
    "109": {
    "title": "Internationalism: Negative",
    "description": "Negative references to international co-operation. Favourable mentions of national independence and sovereignty with regard to their country’s foreign policy, isolation and/or unilateralism as opposed to internationalism.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "110": {
    "title": "European Community/Union: Negative",
    "description": "Negative references to the European Community/Union. May include: Opposition to specific European policies which are preferred by European authorities; Opposition to the net-contribution of their country to the EU budget.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "108_a": {
    "title": "United States: Positive",
    "description": "Favourable mentions of the United States in general.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "110_a": {
    "title": "United States: Negative",
    "description": "Negative references to the United States. May include opposition to specific United States policies which are preferred by United States authorities.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "108_b": {
    "title": "Russia/USSR/CIS: Positive",
    "description": "Favourable mentions of Russia in general.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "110_b": {
    "title": "Russia/USSR/CIS: Negative",
    "description": "Negative references to Russia. May include opposition to specific Russian policies which are preferred by Russian authorities.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "108_c": {
    "title": "China/PRC: Positive",
    "description": "Favourable mentions of China in general.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "110_c": {
    "title": "China/PRC: Negative",
    "description": "Negative references to China. May include opposition to specific Chinese policies which are preferred by Chinese authorities.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "201": {
    "title": "Freedom and Human Rights",
    "description": "Favourable mentions of importance of personal freedom and civil rights in their country and other countries. May include mentions of: The right to the freedom of speech, press, assembly etc.; Freedom from state coercion in the political and economic spheres; Freedom from bureaucratic control; The idea of individualism.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "202": {
    "title": "Democracy",
    "description": "Favourable mentions of democracy as the “only game in town”. General support for their country’s democracy. May also include: Democracy as method or goal in national, international or other organisations (e.g. labour unions, political parties etc.); The need for the involvement of all citizens in political decision-making; Support for either direct or representative democracy; Support for parts of democratic regimes (rule of law, division of powers, independence of courts etc.).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "203": {
    "title": "Constitutionalism: Positive",
    "description": "Support for maintaining the status quo of the constitution. Support for specific aspects of their country’s constitution. The use of constitutionalism as an argument for any policy.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "204": {
    "title": "Constitutionalism: Negative",
    "description": "Opposition to the entirety or specific aspects of their country’s constitution. Calls for constitutional amendments or changes. May include calls to abolish or rewrite the current constitution.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "301": {
    "title": "Federalism",
    "description": "Support for federalism or decentralisation of political and/or economic power. May include: Favourable mentions of the territorial subsidiary principle; More autonomy for any sub-national level in policy making and/or economics; Support for the continuation and importance of local and regional customs and symbols and/or deference to local expertise; Favourable mentions of special consideration for sub-national areas.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "302": {
    "title": "Centralisation",
    "description": "General opposition to political decision-making at lower political levels. Support for unitary government and for more centralisation in political and administrative procedures.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "303": {
    "title": "Governmental and Administrative Efficiency",
    "description": "Need for efficiency and economy in government and administration and/or the general appeal to make the process of government and administration cheaper and more efficient. May include: Restructuring the civil service; Cutting down on the civil service; Improving bureaucratic procedures.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "304": {
    "title": "Political Corruption",
    "description": "Need to eliminate political corruption and associated abuses of political and/or bureaucratic power.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "305": {
    "title": "Political Authority",
    "description": "References to their party’s competence to govern and/or other party’s lack of such competence. Also includes favourable mentions of the desirability of a strong and/or stable government in general.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "401": {
    "title": "Free Market Economy",
    "description": "Favourable mentions of the free market and free market capitalism as an economic model. May include favourable references to: Laissez-faire economy; Superiority of individual enterprise over state and control systems; Private property rights; Personal enterprise and initiative; Need for unhampered individual enterprises.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "402": {
    "title": "Incentives",
    "description": "Favourable mentions of supply side oriented economic policies (assistance to businesses rather than consumers). May include: Financial and other incentives such as subsidies, tax breaks etc.; Wage and tax policies to induce enterprise; Encouragement to start enterprises.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "403": {
    "title": "Market Regulation",
    "description": "Support for policies designed to create a fair and open economic market. May include: Calls for increased consumer protection; Increasing economic competition by preventing monopolies and other actions disrupting the functioning of the market; Defence of small businesses against disruptive powers of big businesses; Social market economy.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "404": {
    "title": "Economic Planning",
    "description": "Favourable mentions of long-standing economic planning by the government. May be: Policy plans, strategies, policy patterns etc.; Of a consultative or indicative nature.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "405": {
    "title": "Corporatism/ Mixed Economy",
    "description": "Favourable mentions of cooperation of government, employers, and trade unions simultaneously. The collaboration of employers and employee organisations in overall economic planning supervised by the state.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "406": {
    "title": "Protectionism: Positive",
    "description": "Favourable mentions of extending or maintaining the protection of internal markets (by their country or other countries). Measures may include: Tariffs; Quota restrictions; Export subsidies.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "407": {
    "title": "Protectionism: Negative",
    "description": "Support for the concept of free trade and open markets. Call for abolishing all means of market protection (in their country or any other country).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "408": {
    "title": "Economic Goals",
    "description": "Broad and general economic goals that are not mentioned in relation to any other category. General economic statements that fail to include any specific goal.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "409": {
    "title": "Keynesian Demand Management",
    "description": "Favourable mentions of demand side oriented economic policies (assistance to consumers rather than businesses). Particularly includes increase private demand through Increasing public demand; Increasing social expenditures. May also include: Stabilisation in the face of depression; Government stimulus plans in the face of economic crises.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "410": {
    "title": "Economic Growth: Positive",
    "description": "The paradigm of economic growth. Includes: General need to encourage or facilitate greater production; Need for the government to take measures to aid economic growth.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "411": {
    "title": "Technology and Infrastructure",
    "description": "Importance of modernisation of industry and updated methods of transport and communication. May include: Importance of science and technological developments in industry; Need for training and research within the economy (This does not imply education in general (see category 506); Calls for public spending on infrastructure such as roads and bridges; Support for public spending on technological infrastructure (e.g.: broadband internet, etc.).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "412": {
    "title": "Controlled Economy",
    "description": "Support for direct government control of economy. May include, for instance: Control over prices; Introduction of minimum wages.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "413": {
    "title": "Nationalisation",
    "description": "Favourable mentions of government ownership of industries, either partial or complete. May also include favourable mentions of government ownership of land.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "414": {
    "title": "Economic Orthodoxy",
    "description": "Need for economically healthy government policy making. May include calls for: Reduction of budget deficits; Retrenchment in crisis; Thrift and savings in the face of economic hardship; Support for traditional economic institutions such as stock market and banking system; Support for strong currency.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "415": {
    "title": "Marxist Analysis: Positive",
    "description": "Positive references to Marxist-Leninist ideology and specific use of Marxist-Leninist terminology by their party (typically but not necessary by communist parties).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "416": {
    "title": "Anti-Growth Economy: Positive",
    "description": "Favourable mentions of anti-growth politics. Rejection of the idea that all growth is good growth. Opposition to growth that causes environmental or societal harm. Call for sustainable economic development.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "501": {
    "title": "Environmental Protection: Positive",
    "description": "General policies in favour of protecting the environment, fighting climate change, and other “green” policies. For instance: General preservation of natural resources; Preservation of countryside, forests, etc.; Protection of national parks; Animal rights. May include a great variance of policies that have the unified goal of environmental protection.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "502": {
    "title": "Culture: Positive",
    "description": "Need for state funding of cultural and leisure facilities including arts and sport. May include: The need to fund museums, art galleries, libraries etc.; The need to encourage cultural mass media and worthwhile leisure activities, such as public sport clubs.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "503": {
    "title": "Equality: Positive",
    "description": "Concept of social justice and the need for fair treatment of all people. This may include: Special protection for underprivileged social groups; Removal of class barriers; Need for fair distribution of resources; The end of discrimination (e.g. racial or sexual discrimination).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "504": {
    "title": "Welfare State Expansion",
    "description": "Favourable mentions of need to introduce, maintain or expand any public social service or social security scheme. This includes, for example, government funding of: Health care; Child care; Elder care and pensions; Social housing.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "505": {
    "title": "Welfare State Limitation",
    "description": "Limiting state expenditures on social services or social security. Favourable mentions of the social subsidiary principle (i.e. private care before state care);",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "506": {
    "title": "Education Expansion",
    "description": "Need to expand and/or improve educational provision at all levels.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "507": {
    "title": "Education Limitation",
    "description": "Limiting state expenditure on education. May include: The introduction or expansion of study fees at all educational levels; Increasing the number of private schools.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "601": {
    "title": "National Way of Life: Positive",
    "description": "Favourable mentions of their country’s nation, history, and general appeals. May include: Support for established national ideas; General appeals to pride of citizenship; Appeals to patriotism; Appeals to nationalism; Suspension of some freedoms in order to protect the state against subversion.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "602": {
    "title": "National Way of Life: Negative",
    "description": "Unfavourable mentions of their country’s nation and history. May include: Opposition to patriotism; Opposition to nationalism; Opposition to the existing national state, national pride, and national ideas.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "603": {
    "title": "Traditional Morality: Positive",
    "description": "Favourable mentions of traditional and/or religious moral values. May include: Prohibition, censorship and suppression of immorality and unseemly behaviour; Maintenance and stability of the traditional family as a value; Support for the role of religious institutions in state and society.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "604": {
    "title": "Traditional Morality: Negative",
    "description": "Opposition to traditional and/or religious moral values. May include: Support for divorce, abortion etc.; General support for modern family composition; Calls for the separation of church and state.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "605": {
    "title": "Law and Order: Positive",
    "description": "Favourable mentions of strict law enforcement, and tougher actions against domestic crime. Only refers to the enforcement of the status quo of their country’s law code. May include: Increasing support and resources for the police; Tougher attitudes in courts; Importance of internal security.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "606": {
    "title": "Civic Mindedness: Positive",
    "description": "Appeals for national solidarity and the need for society to see itself as united. Calls for solidarity with and help for fellow people, familiar and unfamiliar. May include: Favourable mention of the civil society; Decrying anti-social attitudes in times of crisis; Appeal for public spiritedness; Support for the public interest.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "607": {
    "title": "Multiculturalism: Positive",
    "description": "Favourable mentions of cultural diversity and cultural plurality within domestic societies. May include the preservation of autonomy of religious, linguistic heritages within the country including special educational provisions.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "608": {
    "title": "Multiculturalism: Negative",
    "description": "The enforcement or encouragement of cultural integration. Appeals for cultural homogeneity in society.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "701": {
    "title": "Labour Groups: Positive",
    "description": "Favourable references to all labour groups, the working class, and unemployed workers in general. Support for trade unions and calls for the good treatment of all employees, including: More jobs; Good working conditions; Fair wages; Pension provisions etc.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "702": {
    "title": "Labour Groups: Negative",
    "description": "Negative references to labour groups and trade unions. May focus specifically on the danger of unions ‘abusing power’.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "703": {
    "title": "Agriculture and Farmers: Positive",
    "description": "Specific policies in favour of agriculture and farmers. Includes all types of agriculture and farming practises. Only statements that have agriculture as the key goal should be included in this category.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "704": {
    "title": "Middle Class and Professional Groups",
    "description": "General favourable references to the middle class. Specifically, statements may include references to: Professional groups, (e.g.: doctors or lawyers); White collar groups, (e.g.: bankers or office employees), Service sector groups (e.g.: IT industry employees); Old and/or new middle class.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "705": {
    "title": "Underprivileged Minority Groups",
    "description": "Very general favourable references to underprivileged minorities who are defined neither in economic nor in demographic terms (e.g. the handicapped, homosexuals, immigrants). Only includes favourable statements that cannot be classified in other categories (e.g. 503, 504, 604 etc.)",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "706": {
    "title": "Non-economic Demographic Groups",
    "description": "General favourable mentions of demographically defined special interest groups of all kinds. They may include: Women; University students; Old, young, or middle aged people. Might include references to assistance to these groups, but only if these do not fall under other categories (e.g. 503 or 504).",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    },
    "000": {
    "title": "No meaningful category applies",
    "description": "Statements not covered by other categories; sentences devoid of any meaning.",
    "result": "True/False", "explanation": "Which part of the summary caused the result?"
    }
}
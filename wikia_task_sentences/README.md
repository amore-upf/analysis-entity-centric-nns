# README for tsv file: *Sentences generated from unique descriptions*


COLUMNS |   Description
------------ | -------------
entity  |	The entity name as provided by the [Friends Central Wikia database](http://friends.wikia.com)<br>See also `friends_entity_names_semeval_wikia.tsv` for the corresponding SemEval names
entity_id  |	The entity id (as in the SemEval data)
n_properties  |	The number of properties used to generate the unique descriptions
pattern_type  |	The pattern type used to generate the sentence (see below)
sentence  |	The generated natural language sentence
speaker  |	The name of the character which utters the sentence <br>(if it is not a specific character, the speaker_id is set to '---')
speaker_id  |	The speaker id (as in the SemEval data) to_resolve<br> The indices of the beginning and end token of the nominal to resolve (both, beginning and end of the range are inclusive, e.g., This person is a paleontologist --> (1,1))
frequency  |	The frequency of the character (NOT ADDED YET)
combination_type  |	The type of combination of properties: 'attributes', 'relations', 'attributes & relations'
combination_id  |	The id of the combination of properties, i.e., unique description <br>The same description can give rise to multiple sentences depending on the pattern and the order of the properties (all the permutations are generated). E.g., 'This person is a man and a paleontologist', 'He is a paleontologist'.

### PATTERN TYPES 
(*Coded in column 'pattern_type'*)

* **np** 
The NP subject is a property and the subject complement are the remaining properties (min: 1; max:2).
    *	Pattern: This \___ is (attribute: a \___ | relation: a \___ of \___) and [(attribute: a \___ | relation: a \___ of \___)].
    *	The pattern is used for descriptions with min 2 and max 3 properties. 
	In this case, by construction of the dataset, none of the two property alone has a unique referent, so we use the indefinite article.	<br>
	* The sentence is not meant to be uttered by a specific character.<br>
	* Nominal to be resolved: property of NP subject (in case of relations: excluding 'of \___')
	> **Examples:**<br>
      > paleontologist, male --> This paleontologist is a man. (to resolve: paleontologist)<br>
	  > paleontologist, male, teacher --> This paleontologist is a man and a teacher. (to resolve: paleontologist)<br>
	  > 	paleontologist, brother of Monica Geller, teacher --> This paleontologist is the brother of Monica Geller and a teacher. (to resolve: paleontologist)

* **person** <br>	The NP subject is 'This person' and the subject complement are the properties (min: 1; max:3).<br>
	* Pattern: This person is (attribute: a \___ | relation: a \___ of \___) [and|,][(attribute: a \___ | relation: a \___ of \___)][and|,][and (attribute: a \___ | relation: the \___ of \___)]. <br>
		In case of 1 property: This person is (attribute: the \___ | relation: the \___ of \___)].
	* The pattern is used for descriptions with max 3 properties. 
	In case of 1 property, this has by construction of the dataset a unique referent, so we use the definite article.<br>
	* The sentence is not meant to be uttered by a specific character.<br>
	* Nominal to be resolved: 'person'<br>
	> **Examples:**<br>
      > paleontologist --> This person is the paleontologist (to resolve: person).<br>
	  > paleontologist, male --> This person is a paleontologist and a man. (to resolve: person)<br>
	  > paleontologist, male, teacher --> This person is a paleontologist, a man and a teacher. (to resolve: person)<br>
	  > paleontologist, brother of Monica Geller, teacher --> This person is a paleontologist, the brother of Monica Geller and a teacher. (to resolve: person)

* **he_she** <br>	
The subject is a third person pronoun, he or she, specifying one of the properties (gender) and the subject complement are the remaining properties (min: 1; max:2).
	* Pattern: (He|She) is (attribute: a \___ | relation: a \___ of \___) [and|,][(attribute: a \___ | relation: a \___ of \___)].	
	* The pattern is used for descriptions with min 2 and max 3 properties and such that one of them specifies the gender of the entity. 
	In this case, by construction of the dataset, none of the two property alone has a unique referent, so we use the indefinite article.	<br>
	* The sentence is not meant to be uttered by a specific character.<br>
	* Nominal to be resolved: 'he'/'she'	<br>
	> **Examples:**<br>
      > paleontologist, male --> He is a paleontologist. (to resolve: he)<br>
	  > paleontologist, male, teacher --> He is a paleontologist and a teacher. (to resolve: he)<br>
	  > paleontologist, male, brother of Monica Geller --> He is a paleontologist and the brother of Monica Geller. (to resolve: he)

* **I** <br>	
The subject is a first person pronoun, I, and the subject complement are the properties (min: 1; max:3).
	* Pattern: I am (attribute: a \___ | relation: a \___ of \___) [and|,][(attribute: a \___ | relation: a \___ of \___)][and|,][and (attribute: a \___ | relation: a \___ of \___)].
		In case of 1 property: I am (attribute: the \___ | relation: the \___ of \___)].
	* The pattern is used for descriptions with max 3 properties. 
	In case of 1 property, this has by construction of the dataset a unique referent, so we use the definite article.<br>
	* The sentence is meant to be uttered by the entity referred to by the description.<br>
	* Nominal to be resolved: 'I'<br>
	> **Examples:**<br>
      > paleontologist --> I am the paleontologist (to resolve: I).<br>
	  > paleontologist, male --> I am a paleontologist and a man. (to resolve: I)<br>
	  > paleontologist, male, teacher --> I am a paleontologist, a man and a teacher. (to resolve: I)<br>
	  > paleontologist, male, brother of Monica Geller --> I am a paleontologist, a man and the brother of Monica Geller. (to resolve: I)

* **my** <br>	The NP subject is a relation containing a first person possessive, which specifies the bearer of the relation; the subject complement are the remaining properties (min: 1; max:2).
	* Pattern: My \___ is (attribute: a \___ | relation: a \___ of \___) [and|,][(attribute: a \___ | relation: a \___ of \___)].
	* The pattern is used for descriptions with with min 2 and max 3 properties and such that one of them is a relation.
	* The sentence is meant to be uttered by the bearer of the relation specifyied by the NP subject referred to by the description.
	* Nominal to be resolved: relation of NP subject (excluding 'of \___')

    > **Examples:**<br>
      > brother of Monica Geller, paleontologist --> My brother is a paleontologist. [said by Monica Geller] (to resolve: brother)<br>
	   > brother of Monica Geller, ex-boyfriend of Rachel Green, paleontologist --> My brother is an ex-boyfriend of Rachel Green and a paleontologist.  [said by Monica Geller] (to resolve: brother)

#### OTHER PATTERN FEATURES
* **a/an**: 'an' when followed by a vowel sound (using the CMT dictionary from nltk)

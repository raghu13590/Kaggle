# Kaggle - Gene Classification

## Project Description
The Project aims to classify genetic variations from clinical literature. You can find more about the challenge from [here](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)

Data is stored in two separate files - training_text(columns separated by '||') and training_variants (',' delimited). 
After adding both the files into a dataframe, the data would look like this,

### Data Description

```
ID                  Text	                          Gene	    Variation	         Class
0   Cyclin-dependent kinases (CDKs) regulate a var...  FAM58A  Truncating Mutations      1
1    Abstract Background  Non-small cell lung canc...     CBL                 W802*      2
2    Abstract Background  Non-small cell lung canc...     CBL                 Q249E      2
3   Recent evidence has demonstrated that acquired...     CBL                 N454D      3
4   Oncogenic mutations in the monomeric Casitas B...     CBL                 L399V      4
5   Oncogenic mutations in the monomeric Casitas B...     CBL                 V391I      4
6   Oncogenic mutations in the monomeric Casitas B...     CBL                 V430M      5
7   CBL is a negative regulator of activated recep...     CBL              Deletion      1
8    Abstract Juvenile myelomonocytic leukemia (JM...     CBL                 Y371H      4
9    Abstract Juvenile myelomonocytic leukemia (JM...     CBL                 C384R      4
10  Oncogenic mutations in the monomeric Casitas B...     CBL                 P395A      4
11  Noonan syndrome is an autosomal dominant conge...     CBL                 K382E      4
12  Noonan syndrome is an autosomal dominant conge...     CBL                 R420Q      4
13  Noonan syndrome is an autosomal dominant conge...     CBL                 C381A      4
14  Oncogenic mutations in the monomeric Casitas B...     CBL                 P428L      5
15  Noonan syndrome is an autosomal dominant conge...     CBL                 D390Y      4
16  To determine if residual cylindrical refractiv...     CBL  Truncating Mutations      1
17  Acquired uniparental disomy (aUPD) is a common...     CBL                 Q367P      4
18  Oncogenic mutations in the monomeric Casitas B...     CBL                 M374V      5
19  Acquired uniparental disomy (aUPD) is a common...     CBL                 Y371S      4
20   Abstract Background  Non-small cell lung canc...     CBL                  H94Y      6
21  Oncogenic mutations in the monomeric Casitas B...     CBL                 C396R      4
22  Oncogenic mutations in the monomeric Casitas B...     CBL                 G375P      4
23  Recent evidence has demonstrated that acquired...     CBL                 S376F      4
24  Recent evidence has demonstrated that acquired...     CBL                 P417A      4
25  Recent evidence has demonstrated that acquired...     CBL                 H398Y      4
26   Abstract N-myristoylation is a common form of...   SHOC2                   S2G      4
27  Heterozygous mutations in the telomerase compo...    TERT                 Y846C      4
28  Sequencing studies have identified many recurr...    TERT                 C228T      7
29  Heterozygous mutations in the telomerase compo...    TERT                 H412Y      4
```
* Text - clinical evidence to classify genetic mutation
* Gene - the gene where this genetic mutation is location
* Variation - aminoacid change for the mutation
* Class (dependent variable) - One of 9 classes the genetic mutation has been classified on.

The idea is to break the 'Text' column into muliple columns filtering out common english words using NLP techniques and running various classification algortihms.

## Prerequisites
* Python 3.5 or above
* Python IDE

## Instructions
* Clone the project
* [Download datasets from here](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data) and include them in the project folder


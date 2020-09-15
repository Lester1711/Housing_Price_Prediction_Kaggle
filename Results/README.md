# Project 2: AMES Housing Data Set Pricing Prediction


## Problem Statement

Quantum Realty Investment is trying to reduce the number of bad purchases made by inaccurate valuation of properties. Through selecting proper regression techniques (Ridge, Lasso or Elastic Net) and testing them on the Ames Iowa housing market, Quantum hopes to identify a production algorithm which can accurately predict housing prices based on a given set of features. Likewise, the firm also aims to understand what features improve or diminishes a property value. Proper price prediction can help reduces losses and bring profit to both Quantum and it's respective shareholders.


## Executive Summary

Quantum Realty Investment is a top US property investment firm with a business model of identifying under-valued houses for purchase and subsequent resale for profit. Part of the new initiatives taken on within the company's strategic plan would be to use data driven methods such as machine learning in order to better identify value purchases in the housing market.

According to internal estimates, 30% of annual transactions turn out to be bad purchases. Reducing these inaccuracies would not only be able to reduce bad buys but make a profitable turnover.

To solve this, Quantum is hoping to leverage on predictive data techniques to identify more accuratety priced properties. Part of testing this process is to use these data science predictions on the Ames housing city housing set in Iowa. This area has been identified as one of the places where heavy development is underway and median incomes are growing.

With the models constructed by the data science team, Quantum is seeking to gather more precise information on value buys in Ames city. With this information, the firm hopes to exercise profitable transactions based on the predictive model.



## Data Dictionary

<b>AMES Data Dictionary (before cleaning)</b>
    
|Feature                        |Type      |Dataset|Description
|:-----------------------       |:---      |:--- |:---
|<b>ID</b>                      |integer   |Train|Identifier
|<b>PID</b>                     |integer   |Train|Identifier
|<b>MS SubClass</b>             |integer   |Train|Identifies the type of dwelling involved in the sale.
|<b>MS Zoning</b>               |object    |Train|Identifies the general zoning classification of the sale.
|<b>Lot Frontage</b>            |float     |Train|Linear feet of street connected to property
|<b>Lot Area</b>                |integer   |Train|Lot size in square feet
|<b>Street</b>                  |object    |Train|Type of road access to property
|<b>Alley</b>                   |object    |Train|Type of alley access to property
|<b>Lot Shape</b>               |object    |Train|General shape of property
|<b>Land Contour</b>            |object    |Train|Flatness of the property
|<b>Utilities</b>               |object    |Train|Type of utilities available
|<b>Lot Config</b>              |object    |Train|Lot configuration
|<b>Land Slope</b>              |object    |Train|Slope of property
|<b>Neighborhood</b>            |object    |Train|Physical locations within Ames city limits
|<b>Condition 1</b>             |object    |Train|Proximity to various conditions
|<b>Condition 2</b>             |object    |Train|Proximity to various conditions (if more than one is present)
|<b>Bldg Type</b>               |object    |Train|Type of dwelling
|<b>House Style</b>             |object    |Train|Style of dwelling
|<b>Overall Qual</b>            |integer   |Train|Rates the overall material and finish of the house
|<b>Overall Cond</b>            |integer   |Train|Rates the overall condition of the house
|<b>Year Built</b>              |integer   |Train|Original construction date
|<b>Year Remod/Add</b>          |integer   |Train|Remodel date
|<b>Roof Style</b>              |object    |Train|Type of roof
|<b>Roof Matl</b>               |object    |Train|Roof material
|<b>Exterior 1st</b>            |object    |Train|Exterior covering on house
|<b>Exterior 2nd</b>            |object    |Train|Exterior covering on house (if more than one material)
|<b>Mas Vnr Type</b>            |object    |Train|Masonry veneer type
|<b>Mas Vnr Area</b>            |float     |Train|Masonry veneer area in square feet
|<b>Exter Qual</b>              |object    |Train|Evaluates the quality of the material on the exterior
|<b>Exter Cond</b>              |object    |Train|Evaluates the present condition of the material on the exterior
|<b>Foundation</b>              |object    |Train|Type of foundation
|<b>Bsmt Qual</b>               |object    |Train|Evaluates the height of the basement
|<b>Bsmt Cond</b>               |object    |Train|Evaluates the general condition of the basement
|<b>Bsmt Exposure</b>           |object    |Train|Refers to walkout or garden level walls
|<b>BsmtFin Type 1</b>          |object    |Train|Rating of basement finished area
|<b>BsmtFin SF 1</b>            |float     |Train|Type 1 finished square feet
|<b>BsmtFin Type 2</b>          |object    |Train|Rating of basement finished area (if multiple types)
|<b>BsmtFin SF 2</b>            |float     |Train|Type 2 finished square feet
|<b>Bsmt Unf SF</b>             |float     |Train|Unfinished square feet of basement area
|<b>Total Bsmt SF</b>           |float     |Train|Total square feet of basement area
|<b>Heating</b>                 |object    |Train|Type of heating
|<b>Heating QC</b>              |object    |Train|Heating quality and condition
|<b>Central Air</b>             |object    |Train|Central air conditioning
|<b>Electrical</b>              |object    |Train|Electrical system
|<b>1st Flr SF</b>              |integer   |Train|First Floor square feet
|<b>2nd Flr SF</b>              |integer   |Train|Second floor square feet
|<b>Low Qual Fin SF</b>         |integer   |Train|Low quality finished square feet (all floors)
|<b>Gr Liv Area </b>            |integer   |Train|Above grade (ground) living area square feet
|<b>Bsmt Full Bath</b>          |float     |Train|Basement full bathrooms
|<b>Bsmt Half Bath</b>          |float     |Train|Basement half bathrooms
|<b>Full Bath</b>               |integer   |Train|Full bathrooms above grade
|<b>Half Bath</b>               |integer   |Train|Half baths above grade
|<b>Bedroom AbvGr</b>           |integer   |Train|Bedrooms above grade (does NOT include basement bedrooms)
|<b>Kitchen AbvGr</b>           |integer   |Train|Kitchens above grade
|<b>Kitchen Qual</b>            |object    |Train|Kitchen quality
|<b>TotRms AbvGrd</b>           |integer   |Train|Total rooms above grade (does not include bathrooms)
|<b>Functional</b>              |object    |Train|Home functionality (Assume typical unless deductions are warranted)
|<b>Fireplaces</b>              |integer   |Train|Number of fireplaces
|<b>Fireplace Qu</b>            |object    |Train|Fireplace quality
|<b>Garage Type</b>             |object    |Train|Garage location
|<b>Garage Yr Blt</b>           |float     |Train|Year garage was built
|<b>Garage Finish</b>           |object    |Train|Interior finish of the garage
|<b>Garage Cars</b>             |float     |Train|Size of garage in car capacity
|<b>Garage Area</b>             |float     |Train|Size of garage in square feet
|<b>Garage Qual</b>             |object    |Train|Garage quality
|<b>Garage Cond</b>             |object    |Train|Garage condition
|<b>Paved Drive</b>             |object    |Train|Paved driveway
|<b>Wood Deck SF</b>            |integer   |Train|Wood deck area in square feet
|<b>Open Porch SF</b>           |integer   |Train|Open porch area in square feet
|<b>Enclosed Porch</b>          |integer   |Train|Enclosed porch area in square feet
|<b>3Ssn Porch</b>              |integer   |Train|Three season porch area in square feet
|<b>Screen Porch</b>            |integer   |Train|Screen porch area in square feet
|<b>Pool Area</b>               |integer   |Train|Pool area in square feet
|<b>Pool QC</b>                 |object    |Train|Pool quality
|<b>Fence</b>                   |object    |Train|Fence quality 
|<b>Misc Feature</b>            |object    |Train|Miscellaneous feature not covered in other categories    
|<b>Misc Val</b>                |integer   |Train|$Value of miscellaneous feature   
|<b>Mo Sold</b>                 |integer   |Train|Month Sold (MM)  
|<b>Yr Sold</b>                 |integer   |Train|Year Sold (YYYY)
|<b>Sale Type</b>               |object    |Train|Type of sale
|<b>SalePrice</b>               |integer   |Train|Sale price of the property



## Conclusion & Recommendations

Based on the predictive analysis, the features that would add most value to a home are:

-Ground floor and Basement Area Size
-Overall external condition and quality of the property
-Proximity to Stone Brook and Northridge neighbourhoods
-Presence of a Garage

Here are the reasons on why the above factors add most value:

-The greater the livable ground size, the more rooms it can accommodate, maximizes value
-Basements for utility, pricing value and also protection. Ames has high tornado risk.
-Most people in Ames commute by car. The average car ownership in Ames is 2 cars per household
-Stone Brook is located a few minutes drive from Iowa State University
-Northridge is famous for senior care,rehab services and facilities
-Intuitive that the property condition is a strong determinant of housing price
-Value of house will always be strongly correlated to the material and finish of the house
-Newer the condition (construction or renovation), the higher premium demanded

Based on the predictive analysis, the features that would hurt the value to a home are:

-Houses in a PUD (Planned Unit Development) Zone
-Building type is a Townhouse
-Proximity to certain neighbourhoods
-Higher the number of Kitchens above grade

Here are the reasons on why the above factors would hurt the value:

-PUD zone housesholds have to pay maintenance dues for common amentities called Home Owners Association (HOA) fees
-Townhouses share common spaces and amenties and have to also subject to Home Owners Association (HOA) fees
-Certain neighbourhoods may be less desirable in terms of location
-An excessive number of kitchens may be deemed unnecessary

Following things that homeowners could improve in their homes to increase the value:

-Refurbish the house exterior (with good quality materials if possible)
-Add on a garage if there isn't or increase the size of the garage
-Furnish the kitchen and ensure it is in good quality
-Expand on the ground living areas if there is additional land area to utilize

Neighborhoods which seem like a good investment:

-Stone Brook. It is located a few minutes drive from Iowa State University
-Northridge. It is famous for senior care,rehab services and facilities

This model is not likely to generalize to other cities because of the absence of more essential data.

To make it more universal and a comparable model to other cities, consider the following:

-Consider the demographic information such as average age groups, race and income levels of population
-Consider the safety and crime levels
-Economic indicators such as the employment and education rates
-Consider the information taken after the Global Financial Crisis (GFC) years to cater for black swan periods


## Links

https://www.census.gov/quickfacts/amescityiowa
https://datausa.io/profile/geo/ames-ia/
http://www.city-data.com/city/Ames-Iowa.html
https://en.wikipedia.org/wiki/Ames,_Iowa



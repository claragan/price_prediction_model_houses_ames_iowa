## Problem Statement
We are a property consultancy firm which aims to help property owners maximize the value (and selling price) of their properties. We aim to identify the features that are most important to predict sales price (which can be done by ranking the coefficients of each of the features in a regression model). This would allow us to provide our clients with a tool that gives quick estimates of their potential selling prices, and help them identify which aspects of their properties they can improve on to enhance their selling prices.

# Background
Property is a common choice of investment for those who wish to earn passive income, and an asset to hold.

"Would you want to know the price of your property?" This question is something for property owners to responsibly consider and take note of, not just when the need arises for one to sell one's property, but also to periodically check the condition & sellability of one's property. Otherwise, in the event where one urgently requires to sell the property due to being cashstrapped, the property may not be able to be sold at it's best price possible.

Hence, this new feature on the property website allows property owners to know if one's property is truly attractive and sellable. And most importantly, the value of one's property to be at the highest possible attainable price.

## Choice of datasets
The datasets that would be used would be:
test.csv: This data contains the test data for your model. This data will be fed into your regression model to make predictions.
train.csv: This data contains all of the training data for your model.

# Process
The code is divided into the following sections: Data Import & Cleaning, Data Analysis & Visualization, Modelling, Prediction Submission, and Conclusions & Recommendations.

# Data Import & Cleaning
All duplicate values were dropped, and typographical error for the year of the garage being built has been rectified to the best possible correct year. Additionally, checks will be conducted on the variables that have null values. This is especially so for the categorical (i.e. nominal & ordinal) data which suggest there is a null value, when it actually indicates the absence of the variable. For example, a null value for the pool variable indicates that there is actually no pool for that property.

## Feature Engineering 
For categorical variables with null values, new columns will be feature engineered to check how many of the properties in the dataset has the said variable. The variables are: masonry veneer, basement, fireplace, garage, pool, and miscellaneous features. New columns were also created for the total number of full bathrooms for the property, total number of half bathrooms for the property, and age of the property. 

Then based on Forward Selection, some variables were selected while the rest were dropped based on judgement call. The judgement is made by dropping variables of little significance to to the sale price of the property. 

# Data Analysis & Visualization
A heatmap was then produced to analyse the correlation between the continuous variables and the taget variable: sale price of the property. Some variables that have notable correlation with the target variable is the overall quality of the property, above grade (ground) living area square feet of the property, overall condition of the property, and age of the property.

## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---| 
|ms_zoning|*object*|train_4.csv & test_4.csv|Identifies the classification of the zone. The classifications are: Agriculture, Commercial, Floating Village Residential, Industrial, Residential High Density, Residential Low Density, Residential Low Density Park and Residential Medium Density. | 
|lot_area|*int*|train_4.csv & test_4.csv|Lot size in square feet.| 
|street|*object*|train_4.csv & test_4.csv|Identifies the type of road access to the property. The types are: Gravel and Paved.| 
|alley|*object*|train_4.csv & test_4.csv|Identifies the type of alley access to the property. The types are: Gravel, Paved or No Alley.| 
|utilities|*object*|train_4.csv & test_4.csv|Type of utilities available from a scale of 1 to 4, with 1 having only 'Electricity', and 4 having 'All Public Utilities (Electricity, Gas, Water, & Sewage).| 
|lot_config|*object*|train_4.csv & test_4.csv|Lot configuration of the properties. The configurations are: Inside lot, Corner lot, Cul-de-sac, Frontage on 2 sides of property, and Frontage on 3 sides of property.| 
|land_slope|*object*|train_4.csv & test_4.csv|Slope of property from a scale of 1 to 3, with 1 being a 'Gentle Slope' and 3 being a Severe Slope.| 
|neighborhood|*object*|train_4.csv & test_4.csv|Properties that are physically located in the neighborhoods within Ames city limits. The neighbourhoods are: Bloomington Heights, Bluestem, Briardale, Brookside, Clear Creek, College Creek, Crawford, Edwards, Gilbert, Greens, Green Hills, Iowa DOT and Rail Road, Landmark, Meadow Village, Mitchell, North Ames, Northridge, Northpark Villa, Northridge Heights, Northwest Ames, Old Town, South & West of Iowa State University, Sawyer, Sawyer West, Somerset, Stone Brook, Timberland, and Veenker.| 
|overall_qual|*int*|train_4.csv & test_4.csv|Rates the overall material and finish of the house from a scale of 1 to 10, with 1 being 'Very Poor' and 10 being 'Excellent'.| 
|overall_cond|*int*|train_4.csv & test_4.csv|Rates the overall condition of the house from a scale of 1 to 10, with 1 being 'Very Poor' and 10 being 'Excellent'.| 
|year_remod_add|*int*|train_4.csv & test_4.csv|Remodel date of the property (same as construction date if no remodeling or additions)| 
|roof_matl|*object*|train_4.csv & test_4.csv|Roof material of the property. They are: Clay or Tile, Standard (Composite) Shingle, Membrane, Metal, Roll, Gravel & Tar, Wood Shakes, and Wood Shingles.| 
|exter_cond|*object*|train_4.csv & test_4.csv|Evaluates the present condition of the material on the exterior from a scale of 1 to 5, with 1 being 'Poor' and 5 being 'Excellent'.|
|foundation|*object*|train_4.csv & test_4.csv|Type of foundation of the property. They are: Brick & Tile, Cinder Block, Poured Contrete, Slab, Stone, and Wood.| 
|bsmt_cond|*object*|train_4.csv & test_4.csv|Evaluates the general condition of the basement from a scale of 1 to 6, with 1 indicating there is no basement and 6 being 'Excellent'.| 
|total_bsmt_sf|*float* & *int*|train_4.csv & test_4.csv|Total square feet of basement area.| 
|gr_liv_area|*int*|train_4.csv & test_4.csv|Above grade (ground) living area square feet.| 
|bedroom_abvgr|*int*|train_4.csv & test_4.csv|Number of bedrooms above grade (does NOT include basement bedrooms)| 
|kitchen_qual|*object*|train_4.csv & test_4.csv|Kitchen quality from a scale of 1 to 5, with 1 being 'Poor' and 5 being 'Excellent'.|  
|garage_area|*float* & *int*|train_4.csv & test_4.csv|Size of garage in square feet| 
|garage_cond|*object*|train_4.csv & test_4.csv|Evaluates the garage condition from a scale of 1 to 6, with 1 indicating there is no garage and 6 being 'Excellent'.|
|paved_drive |*object*|train_4.csv & test_4.csv|Evaluates the paved driveway from a scale of 1 to 3, with 1 indicating it is 'Dirt/Gravel', and 3 being 'Paved'.| 
|sale_type|*object*|train_4.csv & test_4.csv|Sale type of the property at the point of sale. The types are: Warranty Deed - Conventional, Warranty Deed - Cash, Warranty Deed - VA Loan. Home just constructed and sold, Court Officer Deed/Estate. Contract 15% Down payment regular terms, Contract Low Down payment and low interest, Contract Low Interest, Contract Low Down, and Other.| 
|saleprice|*int*|train_4.csv|Saleprice of the property in dollars.| 
|got_mas_vnr|*int*|train_4.csv & test_4.csv|Determines if the property has a masonry veneer or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|got_bsmt|*int*|train_4.csv & test_4.csv|Determines if the property has basement(s) or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|got_fireplaces|*int*|train_4.csv & test_4.csv|Determines if the property has fireplace(s) or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|got_pool|*int*|train_4.csv & test_4.csv|Determines if the property has pool(s) or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|got_misc|*int*|train_4.csv & test_4.csv|Determines if the property has miscellaneous features (that are not covered in other categories) or not, with 1 indicating there is. Otherwise, it is indicated as 0. Miscellaneous featuress such as an Elevator, 2nd Garage, Shed, Tennis Court, or others.|  
|got_alley|*int*|train_4.csv & test_4.csv|Determines if the property has alley(s) or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|got_fence|*int*|train_4.csv & test_4.csv|Determines if the property has fences or not, with 1 indicating there is. Otherwise, it is indicated as 0.| 
|full_bath_total|*float* or *int*|train_4.csv & test_4.csv|Evaluates the total number of full bathrooms in the property.| 
|half_bath_total|*float* & *int*|train_4.csv & test_4.csv|Evaluates the total number of half bathrooms in the property.| 
|age|*int*|train_4.csv & test_4.csv|Age of the Property, which is determined by the year it was sold minus the year it was built.| 
|num_of_floors|*object*|train_4.csv & test_4.csv|Number of furnished floors the property has. Otherwise, it is indicated as 0.| 

# Modelling
The models LinearRegression, LassoCV and RidgeCV were evaluated. Among this 3 models, Ridge Regression appeared to be the best performing model as it has the highest cross validation score ammong the 3 regression models. 

Since the Ridge Regression shrinks the coefficients of the weaker variables to 0, these variables were dropped and the model was re-ran to improve the model performance. 


## Summary of Analysis
The top 5 variables that affect the saleprices of properties are the Neighborhood, Above Grade Ground Living Area, Sale Type, Overall Quality of the property and the Total Basement Square Feet of the property. This can be seen from the coefficients from the Ridge Regression Model.


## Conclusion & Recommendations
It is obvious that factors such as the size of the property, and quality of the property would have a positive correlation with the sale price of the property. However, through this analysis, property buyers & sellers will be able to make wiser & more informed decisions with regards to their properties. Property sellers can consider remodeling their properties to increase the living area & basement of their property, while property buyers can be more cautious in their choice of property by taking into account the location of the property, and the overall quality & condition of the property. 

Going forward, as a property consultancy firm, we can work together with the local housing development board, and private architecture firms to ensure that new properties will be built with features that meets the market demand and that the properties & neighborhood remain desirable & attractive through the collective efforts with the locals, town counsel, mall owners, Ames IA tourism board. 

To property sellers, they should emphasize on the unique selling point of the property during the process of selling their property. From the interesting locations such as residential floating villages, to freshly built & bought properties, proepty sellers should sell their properties out with their most attractive features which could potentiall offset the undesirable qualities of the properties, if any, to property buyers.
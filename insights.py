import json

cat_col = {
    "device_type": "Most users access the platform via mobile devices, indicating mobile dominance in user interactions.",
    "product_category": "Products are fairly distributed across categories, with toys and beauty slightly more popular, suggesting balanced demand.",
    "shipping_method": "Standard shipping is used significantly more than express and same-day, indicating customers prefer cost-effective delivery options.",
    "payment_method": "Payment methods are relatively evenly distributed, with debit and credit cards being slightly more preferred than others.",
    "used_coupon": "A large proportion of customers use coupons, indicating discounts play a significant role in purchasing behavior.",
    "returned": "The dataset is relatively balanced between returned and non-returned orders, making it suitable for classification without severe imbalance issues."
}


kdeplot = {
    "customer_age": "Customer age is widely distributed with higher density in middle-aged and older groups, indicating a diverse customer base with a slight skew towards mature users.",

    "product_price": "Product prices are heavily right-skewed, with most products priced at lower ranges and very few high-priced items.",

     "discount_percent": (
        "The distribution is right-skewed with a multimodal pattern, showing peaks around 30 percent and 70 to 75 percent discount ranges. "
        "Higher discounts (60 to 75%) are most frequently offered, suggesting an aggressive discounting strategy. "
        "Very low discounts (0 to 5%) are rarely given, indicating the business avoids minimal discount offerings. "
        "The sharp drop after 75 percent suggests discounts beyond that range are uncommon or capped by business policy."
    ),

    "product_rating": (
        "The distribution is roughly bell-shaped and multimodal with peaks around 1, 2.3, and 3.2, spanning 0.5 to 5. "
        "Ratings are fairly evenly spread between 1 and 4.5, indicating diverse customer satisfaction levels. "
        "Very high ratings (above 4.5) and very low ratings (below 0.7) are rare in the dataset. "
        "The multimodal nature suggests distinct customer segments with different satisfaction experiences."
    ),


    "past_purchase_count": (
        "The distribution is highly concentrated with a dominant spike at 9, reaching nearly 19,000 counts. "
        "Most customers have made between 6 and 12 past purchases, indicating a moderately loyal customer base. "
        "Very few customers have more than 15 purchases, showing high repeat buying is relatively uncommon. "
        "The discrete spike pattern suggests past_purchase_count is an integer column with uneven purchase frequencies."
    ),

    "past_return_rate": (
        "The distribution is right-skewed with a peak around 0.15 to 0.20, indicating most customers rarely return products. "
        "Majority of customers have a return rate below 0.4, reflecting generally satisfactory purchase experiences. "
        "A small proportion of customers have high return rates (above 0.6), possibly indicating habitual returners. "
        "The gradual tail towards 0.9 suggests very high return rates are rare but present in the dataset."
    ),

    "delivery_delay_days": "Delivery delays are centered around zero, indicating most orders are delivered on time, with both early and delayed deliveries present.",

    "session_length_minutes": (
        "The distribution is multimodal and right-skewed with peaks around 20, 50, 80, and 120 minutes. "
        "Session lengths are broadly spread from 0 to 135 minutes, indicating highly varied browsing behaviors. "
        "Longer sessions (80 to 130 minutes) are more frequent, suggesting customers tend to spend considerable time browsing. "
        "Very short sessions (under 10 minutes) are least common, implying most customers engage meaningfully with the platform."
    ),

    "num_product_views": "Number of product views is right-skewed, with most users viewing fewer products and a smaller group exploring extensively."
}


boxplot= {
    "customer_age": "Customer age is moderately spread with most values between middle-age groups and no significant outliers, indicating a balanced age distribution.",

    "product_price": "Product price shows strong right skew with numerous high-value outliers, indicating most products are low-priced but a few are significantly expensive.",

    "discount_percent": "Discount percentages are fairly evenly distributed with no extreme outliers, suggesting consistent discount strategies across products.",

    "product_rating": "Product ratings are concentrated in mid to high range with a stable distribution and no major outliers, indicating generally positive feedback.",

    "past_purchase_count": "Most customers have a moderate purchase count, but several high-value outliers exist, indicating a small group of highly active buyers.",

    "past_return_rate": "Past return rate is right-skewed with many high-value outliers, indicating most customers return rarely while a few frequently return products.",

    "delivery_delay_days": "Delivery delays are centered around zero with outliers on both sides, showing most deliveries are on time but some are significantly early or delayed.",

    "session_length_minutes": "Session length is widely spread with no significant outliers, indicating consistent browsing behavior among users.",

    "num_product_views": "Number of product views shows moderate spread without extreme outliers, suggesting typical browsing patterns with some variation."
}

cat_targ = {

    "device_type vs returned": (
        "Mobile users have the lowest return rate (~45%) compared to desktop and tablet users. "
        "Desktop users show the highest return rate meaning slightly more desktop purchases are returned. "
        "Tablet users show a near perfect 50/50 split indicating neutral return behavior. "
        "Overall, device_type has negligible discriminative power over product return likelihood."
    ),

    "product_category vs returned": (
        "Electronics has the lowest return rate (~40%), suggesting customers are most satisfied with electronics purchases. "
        "Clothing has the highest return rate (~55%), meaning more than half of clothing purchases are returned. "
        "Home and sports categories show moderate return rates."
        "Product category shows mild influence on return behavior, with clothing and electronics showing the most contrast."
    ),

    "shipping_method vs returned": (
        "Same_day shipping has the lowest return rate (~42%), suggesting faster delivery leads to better satisfaction. "
        "Express shipping has the higest return rate (~55%), meaning express purchases are returned more frequently. "
        "Standard shipping shows a moderate return rate (~48%), close to same_day shipping. "
        "Shipping method shows a noticeable influence on return behavior, with express shipping driving the most returns."
    ),

    "payment_method vs returned": (
        "Debit card users have the lowest return rate (~45%), indicating more committed purchase decisions. "
        "Apple Pay users show the highest return rate (~50%), with almost equal split between returned and not returned. "
        "Credit card and PayPal users show moderate return rates. "
        "Overall, payment_method shows very weak discriminative power over product return behavior."
    ),

    "used_coupon vs returned": (
        "Customers who did NOT use a coupon (0) show a lower return rate compared to coupon users (1). "
        "Coupon users have a higher return rate, suggesting coupon-driven purchases are slightly more impulsive. "
        "Non-coupon purchases reflect more intentional buying behavior with fewer returns. "
        "used_coupon is a weak but meaningful predictor of return behavior in this dataset."
    )

}

num_targ={
    "customer_age vs returned": (
        "Customers who returned products tend to be slightly older based on the median age difference. "
        "Both groups show a wide age range from around 18 to 80, indicating returns happen across all age groups. "
        "The interquartile range is similar for both classes, suggesting age alone is not a strong predictor of returns. "
        "A mild trend exists where older customers may be slightly more likely to return products."
    ),

    "past_return_rate vs returned": (
        "Customers who returned (1) show a noticeably higher past_return_rate compared to non-returners (0). "
        "This makes it the strongest numerical predictor of the target variable among all features. "
        "The median past_return_rate for returned=1 is visibly higher, confirming habitual return behavior. "
        "This column should be treated as a high-importance feature in any predictive model."
    ),

    "delivery_delay_days vs returned": (
        "Returned products show a wider spread in delivery delay days compared to non-returned ones. "
        "Longer delivery delays appear associated with higher return rates, which is logically consistent. "
        "The presence of extreme outliers in both classes suggests occasional very high or very low delays. "
        "Delivery delay could be a contributing but not decisive factor in product returns."
    ),
}


only_limited_columns={
    "past_purchase_count vs past_return_rate vs returned":
        "The scatter plot reveals no meaningful class separation — returned and non-returned orders are uniformly spread across both features. The marginal KDE curves for both classes almost completely overlap in both dimensions, suggesting that past_purchase_count and past_return_rate are weak standalone predictors of return behavior.",

    "product_price vs discount_percent vs returned":
        "Returned and non-returned items share nearly identical discount distributions, meaning discount level alone does not meaningfully separate the two classes. Most returns concentrate in the low price / high discount zone, which also happens to be where overall purchase volume is highest, making it difficult to isolate return-specific behavior from general purchase trends.",

    "session_length_minutes vs num_product_views vs returned":
        "The scatter plot shows no spatial separation between returned and non-returned orders, and the KDE curves for both classes are nearly identical — indicating neither feature is a strong standalone predictor.",

    "used_coupon vs payment_method vs returned":
        "The relationship between coupon usage and return rate varies noticeably across payment methods. Apple Pay users who used a coupon show a higher return rate compared to those who did not, while debit card users exhibit the opposite trend. PayPal transactions show a roughly balanced return rate regardless of coupon usage, and credit card purchases with coupons have near-equal return rates for both classes. This suggests that the effect of coupon usage on returns is payment-method-dependent.",

    "device_type vs shipping_method vs returned":
        "Mobile users account for the largest share of purchases across all shipping methods, particularly under standard shipping. For mobile + standard shipping, non-returned orders significantly outnumber returned ones, indicating a lower return risk in this combination. Conversely, desktop users with express shipping show a return rate that exceeds non-returns, making it a relatively higher-risk segment. Same-day shipping has very low volume across all device types, making its patterns less statistically reliable, while tablet users display consistently balanced return rates with no strong directional signal.",

    "product_category vs shipping_method vs returned":
        "Toys is the highest-volume category under both standard and express shipping, with return rates close to non-return rates overall. However, toys combined with express shipping stand out as a high return-risk combination, where returned orders visibly exceed non-returned ones. Electronics under standard shipping shows the opposite trend, with buyers more likely to keep their purchases. Beauty products also show elevated returns under express shipping, while sports products have relatively fewer returns across all methods. Clothing returns are slightly elevated or balanced, which aligns with the expected uncertainty around fit and sizing in online shopping."

}

all_numerical_columns="The correlation heatmap reveals that all numerical features in the dataset are very weakly correlated with each other, with all pairwise correlation values falling well below 0.1 in absolute terms.This near-zero correlation across all feature pairs confirms the absence of multicollinearity, which is favorable for linear models, but also suggests that no single numerical feature carries a strong linear relationship with another. Overall, the features appear to be largely independent of one another."
all_categorical_columns=''

multivariate={'all_numerical_columns':all_numerical_columns,'all_categorical_columns':all_categorical_columns,'only_limited_columns':only_limited_columns}

bivariate={'cat_targ':cat_targ,'num_targ':num_targ}

num_col={'kdeplot':kdeplot,'boxplot':boxplot}
univariate={'cat_col':cat_col,'num_col':num_col}
insights={'univariate':univariate,'bivariate':bivariate,'multivariate':multivariate}
with open("eda_insights.json", "w") as f:
    json.dump(insights, f, indent=4)
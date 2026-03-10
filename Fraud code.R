# --- PROETOIMASIA KAI FORTOSI ERGALEION ---
# Ksekinao egkathistontas ta aparaitita paketa gia analysi kai grafimata
install.packages("readxl") 
install.packages("dplyr")
install.packages("ggplot2")
library("readxl")
library("dplyr")
library("ggplot2")

# Diavazo to arxeio Excel me ta dedomena tis apatis
data <- read_excel("insurance fraud data.xlsx")
data <- as.data.frame(data)

# Metatrepo tis stiles se arithmitikes gia na mboro na kano prakseis kai montela
data$year_as_customer <- as.numeric(data$year_as_customer)
data$fraud_reported <- as.numeric(data$fraud_reported)
data$policy_premium <- as.numeric(data$policy_premium)
data$P_property_damage <- as.numeric(data$P_property_damage)

# Elenxo ti domi ton dedomenon mou gia na vevaiotho oti ola einai ok
str(data)

# Fortono pio exeidikevmena paketa gia Machine Learning kai diaxeirisi imbalanced dedomenon
install.packages("bnlearn")
install.packages("imbalance")
install.packages("sparklyr")
library("bnlearn")
library("imbalance")
library("sparklyr")

##################### OPTIKOPOIISI TOU PROVLIMATOS ##################
# Edo ftiaxno ena barplot gia na deixo poso liga einai ta peristatika apatis (Fraud) 
# se sxesi me ta kanonika. Ayto einai to vasiko mou provlima (Class Imbalance).
fraud_summary <- table(data$fraud_reported)
fraud_summary <- as.data.frame(fraud_summary)
colnames(fraud_summary) <- c("Fraudulent", "Count")
fraud_summary$Fraudulent <- ifelse(fraud_summary$Fraudulent == 1, "Fraudulent", "Non-Fraudulent")

ggplot(fraud_summary, aes(x = Fraudulent, y = Count, fill = Fraudulent)) +
  geom_bar(stat = "identity") +
  labs(title = "Sygrisi Peristatikon Apatis vs Kanonikon",
       x = "Typos Peristatikou",
       y = "Plithos") +
  theme_minimal() +
  scale_fill_manual(values = c("Non-Fraudulent" = "blue", "Fraudulent" = "red"))

########################## PERIGRAFIKI ANALYSI (EDA) #############################

# Psaxno na do se poies polis exoume ta perissotera krousmata apatis
fraud_city_summary <- data %>%
  filter(fraud_reported == 1) %>%
  group_by(policy_state) %>%
  summarise(fraud_summary = n()) %>%
  arrange(desc(fraud_summary)) %>%
  slice_head(n = 5) 

ggplot(fraud_city_summary, aes(x = reorder(policy_state, -fraud_summary), y = fraud_summary, fill = policy_state)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 5 Politeies me ta perissotera krousmata Apatis",
       x = "Politeia",
       y = "Arithmos Krousmaton") +
  theme_minimal()

# Analyo to fylo ton asfalismenon gia na do an yparxei kapoia tasi ekei
sex_summary <- data %>%
  group_by(insured_sex) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(percentage = count / sum(count) * 100)

ggplot(sex_summary, aes(x = "", y = count, fill = as.factor(insured_sex))) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  labs(title = "Katanomi Fylou sto Deigma", fill = "Fylo") +
  theme_void() +
  scale_fill_manual(values = c("0" = "blue", "1" = "pink"), labels = c("Andras", "Gynaika"))

############################## EXISORROPISI DEDOMENON (BALANCING) #####################
# Xorizo ta dedomena se Train (85%) kai Test (15%) gia na ekpaidefso to montelo
numerical_data <- data %>% select_if(is.numeric)
set.seed(2)
train_sample_ids <- base::sample(seq_len(nrow(numerical_data)), size = floor(0.85 * nrow(numerical_data)))
train_df <- numerical_data[train_sample_ids, ]
test_df <- numerical_data[-train_sample_ids, ]

# Efarmozo ti methodo SMOTE gia na "genniso" texnita dedomena apatis. 
# Xoris ayto, to montelo tha "agnoouse" tin apati epeidi einai poly spania.
smote_train_df <- train_df %>%
  mutate(fraud_reported = factor(fraud_reported)) %>%
  oversample(ratio = 0.99, method = "SMOTE", classAttr = "fraud_reported") %>%
  mutate(fraud_reported = as.integer(as.character(fraud_reported)))

############################## EKPAIDEFSI MONTELOU (LightGBM) ########################
# Ekpaidefo to montelo LightGBM dyo fores: mia me ta arxika dedomena kai mia me ta SMOTE
# gia na sygrino ti diafora stin apodosi.
library("lightgbm")
label_col <- which(names(train_df) == "fraud_reported")
test_mtx <- as.matrix(test_df)
test_x <- test_mtx[, -label_col]
test_y <- test_mtx[, label_col]

params <- list(objective = "binary", learning_rate = 0.05)

# Ekpaidefsi me SMOTE dedomena (exisorropimena)
smote_train_mtx <- as.matrix(smote_train_df)
smote_train_data <- lgb.Dataset(smote_train_mtx[, -label_col], label = smote_train_mtx[, label_col])
smote_model <- lgb.train(data = smote_train_data, params = params, nrounds = 300L)

# Vlepo poia xaraktiristika (Features) einai ta pio simantika gia tin provlepsi apatis
smote_imp <- lgb.importance(smote_model, percentage = TRUE)
ggplot(smote_imp, aes(x = Frequency, y = reorder(Feature, Frequency), fill = Frequency)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low="steelblue", high="tomato") +
  labs(title = "Poioi paragontes prodidoun tin apati;")

########################### AXIOLOGISI (Confusion Matrix & ROC) ############################
# Ftiaxno ta Confusion Matrices gia na do poses fores pesame mesa kai poses ekso
library(caret)
smote_preds <- predict(smote_model, test_mtx[, -label_col])
binary_smote_preds <- ifelse(smote_preds > 0.5, 1, 0)

# Telos, sxediazo tin kampyli ROC gia na metro tin akriveia tou montelou (AUC)
library(PRROC)
smote_fg <- smote_preds[test_df$fraud_reported == 1]
smote_bg <- smote_preds[test_df$fraud_reported == 0]
smote_roc <- roc.curve(scores.class0 = smote_fg, scores.class1 = smote_bg, curve = TRUE)
plot(smote_roc)
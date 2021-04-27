SELECT provider.id
     , provider.provider_name
     , provider.provider_key
     , provider.type
     , provider.trading_name
     , provider_account_key
     , provider_account.account_name
     , provider_account.is_inactive
     , provider_account.id account_id
  FROM HH_provider_account provider_account
  INNER JOIN HH_provider provider
    ON provider.id = provider_account.provider_id
 WHERE provider_account.provider_id IN(SELECT DISTINCT invoice.provider_account_id
										 FROM HH_invoice invoice
										WHERE YEAR(invoice.created_at) = 2021
                                      )
  ORDER BY provider.provider_name
         , provider.trading_name
         , provider_id
      

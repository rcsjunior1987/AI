SELECT member.id
	 , member.member_key
     , member.date_of_birth
     , member.first_name
     , member.last_name
     , price_zone_code
  FROM SNOW_csm_consumer_user csm_user 
  INNER JOIN HH_member member ON member.membership_number = csm_user.u_ndis_number
 WHERE csm_user.u_stage = 'li_managed'
 
	AND member.id IN (SELECT DISTINCT invoice.member_id
						FROM HH_invoice invoice
          
						INNER JOIN HH_provider_account provider_account
								ON provider_account.id = invoice.provider_account_id
          
						INNER JOIN HH_provider provider
							    ON provider.id = provider_account.provider_id
          
						LEFT JOIN HH_claim claim								
							   ON claim.invoice_id = invoice.id
         
					   WHERE YEAR(invoice.created_at) = 2021
						 AND YEAR(claim.created_at) = 2021
   
					 )
 
  ORDER BY member.first_name
         , member.last_name
         , member.date_of_birth
 